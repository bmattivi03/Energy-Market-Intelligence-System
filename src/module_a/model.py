"""Multi-scale LSTM for probabilistic DE-LU load forecasting (Module A).

Architecture
------------
Two parallel LSTM branches encode different temporal scales, then a shared
MLP head maps the concatenated hidden states to 24-horizon × 3-quantile output.

    x_short (B, 48, F)  → LSTM(128, 2 layers) → h_short (B, 128)  ─┐
                                                                      ├→ cat(256) → MLP → (B, 24, 3)
    x_long  (B, 168, F) → LSTM(128, 1 layer)  → h_long  (B, 128)  ─┘

Loss: mean pinball loss across horizon and quantile dims.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from data.splits import HORIZON, LOOKBACK
from module_a.features import TARGET_COL
from utils.reproducibility import select_device

# ---------------------------------------------------------------- constants

SHORT_WINDOW = 48        # 2-day short-scale branch input
LONG_WINDOW  = LOOKBACK  # 168h = 1-week long-scale branch input
QUANTILES    = (0.1, 0.5, 0.9)

# ---------------------------------------------------------------- dataset


class LoadSequenceDataset(Dataset):
    """Sliding-window dataset over a scaled feature array.

    Each item:
        x_short : float32 tensor (SHORT_WINDOW, n_features)
        x_long  : float32 tensor (LONG_WINDOW,  n_features)
        y       : float32 tensor (HORIZON,)   — load at origin+1h..origin+24h
    """

    def __init__(
        self,
        features: np.ndarray,   # (T, F) scaled
        targets:  np.ndarray,   # (T,)   scaled load
        *,
        stride: int = 1,
    ):
        self.features = features.astype(np.float32)
        self.targets  = targets.astype(np.float32)
        T = len(features)
        # Valid origins: need LONG_WINDOW history + HORIZON future
        self.origins = list(range(LONG_WINDOW, T - HORIZON, stride))

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, idx: int):
        t = self.origins[idx]
        x_short = self.features[t - SHORT_WINDOW : t]
        x_long  = self.features[t - LONG_WINDOW  : t]
        y       = self.targets[t : t + HORIZON]
        return (
            torch.from_numpy(x_short),
            torch.from_numpy(x_long),
            torch.from_numpy(y),
        )


# ---------------------------------------------------------------- loss


def pinball_loss(
    pred: torch.Tensor,    # (B, H, Q)
    target: torch.Tensor,  # (B, H)
    quantiles: Sequence[float] = QUANTILES,
) -> torch.Tensor:
    q = torch.tensor(quantiles, dtype=torch.float32, device=pred.device)
    target_exp = target.unsqueeze(-1)          # (B, H, 1)
    err = target_exp - pred                    # (B, H, Q)
    loss = torch.where(err >= 0, q * err, (q - 1) * err)
    return loss.mean()


# ---------------------------------------------------------------- model


class MultiScaleLSTM(nn.Module):
    """Two-branch LSTM encoder + shared quantile head."""

    def __init__(
        self,
        n_features: int,
        *,
        hidden: int = 128,
        dropout: float = 0.2,
        num_layers_short: int = 2,
        num_layers_long: int = 1,
        horizon: int = HORIZON,
        n_quantiles: int = len(QUANTILES),
    ):
        super().__init__()
        self.horizon     = horizon
        self.n_quantiles = n_quantiles

        self.lstm_short = nn.LSTM(
            n_features, hidden,
            num_layers=num_layers_short, batch_first=True,
            dropout=dropout if num_layers_short > 1 else 0.0,
        )
        self.lstm_long = nn.LSTM(
            n_features, hidden,
            num_layers=num_layers_long, batch_first=True,
            dropout=dropout if num_layers_long > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon * n_quantiles),
        )

    def forward(
        self,
        x_short: torch.Tensor,  # (B, 48, F)
        x_long:  torch.Tensor,  # (B, 168, F)
    ) -> torch.Tensor:          # (B, H, Q)
        _, (h_short, _) = self.lstm_short(x_short)
        _, (h_long,  _) = self.lstm_long(x_long)
        h = torch.cat([h_short[-1], h_long[-1]], dim=-1)  # (B, 256)
        out = self.head(h)                                  # (B, H*Q)
        return out.view(-1, self.horizon, self.n_quantiles)


# ---------------------------------------------------------------- forecaster


def _select_device() -> torch.device:
    """Thin wrapper over the canonical ``select_device`` (honours EMIS_FORCE_CPU)."""
    return torch.device(select_device())


class MultiScaleLSTMForecaster:
    """Sklearn-style wrapper around MultiScaleLSTM.

    Usage::

        forecaster = MultiScaleLSTMForecaster()
        forecaster.fit(train_feat_df, val_feat_df=val_feat_df)
        preds = forecaster.predict_quantiles(test_feat_df)
    """

    def __init__(
        self,
        *,
        hidden: int = 128,
        dropout: float = 0.2,
        num_layers_short: int = 2,
        num_layers_long: int = 1,
        quantiles: tuple[float, ...] = QUANTILES,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 20,
        random_state: int = 42,
        checkpoint_path: Optional[Path] = None,
    ):
        self.hidden            = hidden
        self.dropout           = dropout
        self.num_layers_short  = num_layers_short
        self.num_layers_long   = num_layers_long
        self.quantiles         = tuple(quantiles)
        self.lr                = lr
        self.weight_decay      = weight_decay
        self.batch_size        = batch_size
        self.max_epochs        = max_epochs
        self.patience          = patience
        self.random_state      = random_state
        self.checkpoint_path   = checkpoint_path

        self._feat_cols: list[str] | None = None
        self._feat_scaler: StandardScaler | None = None
        self._tgt_scaler: StandardScaler | None = None
        self._model: MultiScaleLSTM | None = None

    # ---- internal helpers

    def _seed(self) -> None:
        import random
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

    def _feature_cols(self, df: pd.DataFrame) -> list[str]:
        """All float columns except target."""
        return [c for c in df.select_dtypes(include=[np.number]).columns
                if c != TARGET_COL]

    def _to_arrays(
        self, df: pd.DataFrame, *, fit_scalers: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (features_scaled, targets_scaled) as float32 arrays."""
        X = df[self._feat_cols].to_numpy(dtype=np.float64)
        y = df[[TARGET_COL]].to_numpy(dtype=np.float64)

        if fit_scalers:
            self._feat_scaler = StandardScaler().fit(X)
            self._tgt_scaler  = StandardScaler().fit(y)

        X_sc = self._feat_scaler.transform(X).astype(np.float32)
        y_sc = self._tgt_scaler.transform(y).astype(np.float32).squeeze(-1)
        # Replace any residual NaN/Inf (imputed data edge-cases) with 0
        X_sc = np.nan_to_num(X_sc, nan=0.0, posinf=0.0, neginf=0.0)
        y_sc = np.nan_to_num(y_sc, nan=0.0)
        return X_sc, y_sc

    # ---- public API

    def fit(
        self,
        feat_df: pd.DataFrame,
        *,
        val_feat_df: Optional[pd.DataFrame] = None,
    ) -> "MultiScaleLSTMForecaster":
        self._seed()
        device = _select_device()
        print(f"Device: {device}")

        self._feat_cols = self._feature_cols(feat_df)
        X_tr, y_tr = self._to_arrays(feat_df, fit_scalers=True)

        train_ds = LoadSequenceDataset(X_tr, y_tr, stride=1)
        train_dl = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        val_dl = None
        if val_feat_df is not None:
            X_va, y_va = self._to_arrays(val_feat_df)
            val_ds = LoadSequenceDataset(X_va, y_va, stride=1)
            val_dl = DataLoader(val_ds, batch_size=self.batch_size * 2,
                                shuffle=False, num_workers=0)

        n_features = X_tr.shape[1]
        model = MultiScaleLSTM(
            n_features, hidden=self.hidden, dropout=self.dropout,
            num_layers_short=self.num_layers_short,
            num_layers_long=self.num_layers_long,
        ).to(device)
        self._model = model

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, min_lr=1e-5,
        )

        best_val_loss = float("inf")
        best_state    = None
        no_improve    = 0
        q_tensor      = torch.tensor(self.quantiles, device=device)

        for epoch in range(1, self.max_epochs + 1):
            # train
            model.train()
            tr_loss = 0.0
            for x_s, x_l, y in train_dl:
                x_s, x_l, y = x_s.to(device), x_l.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(x_s, x_l)
                loss = pinball_loss(pred, y, self.quantiles)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tr_loss += loss.item() * len(y)
            tr_loss /= len(train_ds)

            # validate
            if val_dl is not None:
                model.eval()
                va_loss = 0.0
                with torch.no_grad():
                    for x_s, x_l, y in val_dl:
                        x_s, x_l, y = x_s.to(device), x_l.to(device), y.to(device)
                        pred = model(x_s, x_l)
                        va_loss += pinball_loss(pred, y, self.quantiles).item() * len(y)
                va_loss /= len(val_ds)
                scheduler.step(va_loss)
                monitor = va_loss
            else:
                monitor = tr_loss

            if epoch % 10 == 0 or epoch == 1:
                val_str = f"  val={monitor:.4f}" if val_dl else ""
                print(f"epoch {epoch:03d}  train={tr_loss:.4f}{val_str}")

            if monitor < best_val_loss - 1e-6:
                best_val_loss = monitor
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"Early stop at epoch {epoch}  best={best_val_loss:.4f}")
                    break

        # restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        if self.checkpoint_path is not None:
            self.save(Path(self.checkpoint_path))

        return self

    def predict_quantiles(
        self,
        feat_df: pd.DataFrame,
        *,
        stride: int = 1,
    ) -> pd.DataFrame:
        """Return quantile forecasts as a long-form DataFrame.

        Columns: q10, q50, q90.
        Index:   (origin_ts, horizon_h) MultiIndex.
        """
        if self._model is None:
            raise RuntimeError("call fit() first")

        device = _select_device()
        self._model.eval()
        self._model.to(device)

        X_sc, y_sc = self._to_arrays(feat_df)
        ds = LoadSequenceDataset(X_sc, y_sc, stride=stride)
        dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

        preds_scaled: list[np.ndarray] = []
        with torch.no_grad():
            for x_s, x_l, _ in dl:
                out = self._model(x_s.to(device), x_l.to(device))  # (B, H, Q)
                preds_scaled.append(out.cpu().numpy())

        preds_scaled_arr = np.concatenate(preds_scaled, axis=0)  # (N, H, Q)

        # Inverse-scale: target scaler was fit on (T, 1); broadcast over H and Q
        mean = self._tgt_scaler.mean_[0]
        std  = self._tgt_scaler.scale_[0]
        preds = preds_scaled_arr * std + mean  # (N, H, Q)

        # Build MultiIndex DataFrame
        origins = [feat_df.index[t] for t in ds.origins]
        rows = []
        for i, origin in enumerate(origins):
            for h in range(1, HORIZON + 1):
                rows.append({
                    "origin_ts": origin,
                    "horizon_h": h,
                    "q10": preds[i, h - 1, 0],
                    "q50": preds[i, h - 1, 1],
                    "q90": preds[i, h - 1, 2],
                })
        result = pd.DataFrame(rows).set_index(["origin_ts", "horizon_h"])
        return result

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state":  self._model.state_dict(),
            "feat_cols":    self._feat_cols,
            "feat_scaler":  self._feat_scaler,
            "tgt_scaler":   self._tgt_scaler,
            "init_kwargs":  {
                "hidden":            self.hidden,
                "dropout":           self.dropout,
                "num_layers_short":  self.num_layers_short,
                "num_layers_long":   self.num_layers_long,
                "quantiles":         self.quantiles,
                "lr":                self.lr,
                "weight_decay":      self.weight_decay,
                "batch_size":        self.batch_size,
                "max_epochs":        self.max_epochs,
                "patience":          self.patience,
                "random_state":      self.random_state,
            },
        }
        torch.save(payload, path)
        print(f"Saved checkpoint: {path}")

    @classmethod
    def load(cls, path: Path) -> "MultiScaleLSTMForecaster":
        payload = torch.load(path, weights_only=False, map_location="cpu")
        obj = cls(**payload["init_kwargs"])
        obj._feat_cols    = payload["feat_cols"]
        obj._feat_scaler  = payload["feat_scaler"]
        obj._tgt_scaler   = payload["tgt_scaler"]
        n_features = len(obj._feat_cols)
        obj._model = MultiScaleLSTM(
            n_features, hidden=obj.hidden, dropout=obj.dropout,
            num_layers_short=obj.num_layers_short,
            num_layers_long=obj.num_layers_long,
        )
        obj._model.load_state_dict(payload["model_state"])
        obj._model.eval()
        return obj
