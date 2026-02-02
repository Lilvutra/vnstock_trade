import pandas as pd
import numpy as np
import os

# This file generates labels after feature engineering step
# labels matter more than features in stock ML!
# Currently have:
# t+N return
# volatility-adjusted return
# quantile-based ranking
# Viable further development: Barrier / event-based labels

def generate_labels(df: pd.DataFrame, hold_period: int = 5, quantiles: int = 5) -> pd.DataFrame:
    """
    Generate t+N return labels based on quantiles and volatility adjustment.

    Parameters:
    - df: DataFrame with at least 'close' price column.
    - hold_period: Number of days to hold for return calculation (default is 5).
    - quantiles: Number of quantiles to divide returns into (default is 5).

    Returns:
    - DataFrame with additional 'return_t+N' and 'label' columns.
    """
    df = df.copy()
   
    # Calculate t+N returns
    # why df['close'] not df[return]? because we need price at time t and t+N to calculate return, df['return'] is calculated from past prices(backwards looking)
    df['return_t+N'] = df['close'].shift(-hold_period) / df['close'] - 1
    print(df)
    
    # Calculate rolling volatility (standard deviation of returns)
    df['volatility'] = df['return_t+N'].rolling(window=hold_period).std()
    print(df)
    
    # If we don't adjust for volatility, the model learns chase wild, noisy stocks, adjusting helps stable opportunities, consistent alpha and risk-aware decisions 
    # If stock A: smooth climb, stock B: wild up-down-up-down, which stock would you trust with real money?
    # Volatility-adjusted returns
    df['vol_adj_return'] = df['return_t+N'] / df['volatility']
    
    # Drop rows with NaN values resulting from shifts and rolling calculations
    df.dropna(subset=['vol_adj_return'], inplace=True)
   
    # Assign quantile-based labels
    df['label'] = pd.qcut(df['vol_adj_return'], q=quantiles, labels=False)
    print(df)
    
    print(f"fr mean: {df.groupby('label')['vol_adj_return'].mean()}")

    
    #return df[['return_t+N', 'vol_adj_return', 'label']]
    return df
    # want to merge df with label with features_VNM

if __name__ == "__main__":
    # Example usage
    data = {
        'close': [100, 102, 101, 105, 107, 110, 108, 111, 115, 117, 120]
    }
    df = pd.DataFrame(data)
    labels_df = generate_labels(df)
    print(f"label_df: {labels_df}")