import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def evaluate_trading_logic(output_dir="data/module_b_outputs_catboost_stacked"):
    # 1. Load data
    results_path = Path(output_dir) / "test.parquet"
    if not results_path.exists():
        print(f"Error: {results_path} not found. Generate predictions first.")
        return
        
    results = pd.read_parquet(results_path)
    actual = pd.read_parquet("data/splits/test.parquet")[["price"]]
    df = results.join(actual).dropna()

    # 2. Add 'previous_price' to calculate trends
    # In our 24h horizon setup, we compare predicted T+h with price at T.
    # To simplify for the evaluation, we'll look at the price 1h before for the test set.
    # Note: This is an approximation of Directional Accuracy.
    df['prev_price'] = df['price'].shift(1)
    df = df.dropna()

    # 3. Directional Accuracy (DA)
    # Real direction: is current price > previous price?
    df['real_dir'] = (df['price'] > df['prev_price']).astype(int)
    # Predicted direction: is predicted price > previous price?
    df['pred_dir'] = (df['q50_price'] > df['prev_price']).astype(int)
    
    da = (df['real_dir'] == df['pred_dir']).mean()

    # 4. Naive Trading Strategy (PnL Simulation)
    # Strategy: 
    # - If pred says UP: Buy at prev_price, sell at price. Profit = price - prev_price
    # - If pred says DOWN: Sell at prev_price (short), buy at price. Profit = prev_price - price
    # (Minus a small transaction cost of 0.1 EUR/MWh)
    cost = 0.1
    df['pnl'] = np.where(df['pred_dir'] == 1, 
                         df['price'] - df['prev_price'] - cost, 
                         df['prev_price'] - df['price'] - cost)
    
    cumulative_pnl = df['pnl'].cumsum()
    total_profit = df['pnl'].sum()
    sharpe = df['pnl'].mean() / df['pnl'].std() * np.sqrt(365 * 24) # Annualized Sharpe approx

    print("\n" + "="*40)
    print("      TRADING-ORIENTED EVALUATION")
    print("="*40)
    print(f"Directional Accuracy: {da:.2%}")
    print(f"Total Naive PnL:     {total_profit:.2f} €/MWh")
    print(f"Avg Profit per hour: {df['pnl'].mean():.4f} €/MWh")
    print(f"Estimated Sharpe:    {sharpe:.2f}")
    
    # 5. Volatility Capture
    # How does DA behave during high volatility?
    vol_threshold = df['price'].std()
    df['is_volatile'] = (df['price'] - df['prev_price']).abs() > vol_threshold
    da_vol = df[df['is_volatile']]['real_dir'] == df[df['is_volatile']]['pred_dir']
    print(f"DA in Volatile Hours: {da_vol.mean():.2%}")

    print("="*40)
    print("Interpretation: If DA > 55% and PnL is positive, the model has \nmarket-beating potential regardless of MAE.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/module_b_outputs_catboost_stacked", help="Prediction directory")
    args = parser.parse_args()
    evaluate_trading_logic(args.dir)
