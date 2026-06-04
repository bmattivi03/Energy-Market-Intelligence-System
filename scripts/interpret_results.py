import pandas as pd
import numpy as np

# Load data
results = pd.read_parquet("data/module_b_outputs/test.parquet")
actual = pd.read_parquet("data/splits/test.parquet")[["price"]]
df = results.join(actual).dropna()

# Metrics
df['error'] = df['price'] - df['q50_price']
df['abs_error'] = df['error'].abs()
df['interval_width'] = df['q90_price'] - df['q10_price']
picp = ((df['price'] >= df['q10_price']) & (df['price'] <= df['q90_price'])).mean()
mae = df['abs_error'].mean()
rmse = np.sqrt((df['error']**2).mean())

print(f"--- Global Performance ---")
print(f"MAE: {mae:.2f} €/MWh")
print(f"RMSE: {rmse:.2f} €/MWh")
print(f"PICP (80% Target): {picp:.2%}")

print(f"\n--- Uncertainty Analysis ---")
print(f"Avg Interval Width: {df['interval_width'].mean():.2f} €/MWh")
print(f"Correlation (Price vs Width): {df['price'].corr(df['interval_width']):.4f}")

# Quantile Bias
print(f"\n--- Quantile Bias ---")
print(f"Under-prediction (Price < q10): { (df['price'] < df['q10_price']).mean():.2%}")
print(f"Over-prediction (Price > q90): { (df['price'] > df['q90_price']).mean():.2%}")
