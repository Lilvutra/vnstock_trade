import os
import numpy as np
import pandas as pd


# Relative Strength Index (RSI) calculation
def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window=window, min_periods=window).mean()
    ma_down = down.rolling(window=window, min_periods=window).mean()
    rs = ma_up / ma_down
    return 100 - 100 / (1 + rs)

def build_features(df: pd.DataFrame,
                   symbol_col: str = 'symbol',
                   time_col: str = 'time',
                   windows=(5, 10, 21),
                   rsi_window: int = 14) -> pd.DataFrame:
    """Compute a standard set of quantitative features for multi-stock daily data.

    Expects columns: time, open, high, low, close, volume, symbol (or custom names).
    Returns the original dataframe with new feature columns.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    def _features(g: pd.DataFrame) -> pd.DataFrame:
        # Simple returns add changes; log returns add growth rates of a multiplicative process
        # We can use both to measure change, but for any aggregation, modeling, or statistics, only log returns are valid

        g = g.sort_values(time_col).copy()
        g['return_1'] = g['close'].pct_change() #simple daily return
        g['log_return_1'] = np.log(g['close']).diff() # log daily return

        for w in windows:
            g[f'ma_{w}'] = g['close'].rolling(w).mean() # moving average over w days
            g[f'return_{w}'] = g['close'].pct_change(w) # simple return over w days
            g[f'volatility_{w}'] = g['log_return_1'].rolling(w).std() # volatility (std dev of log returns) over w days

        # Lagged returns
        for lag in range(1, 6):
            g[f'lag_logret_{lag}'] = g['log_return_1'].shift(lag)

        # Momentum and volume features
        g['mom_5'] = g['close'] / g['close'].shift(5) - 1
        g['vol_change'] = g['volume'].pct_change() # pct = percentage change
        g['vol_ma_5'] = g['volume'].rolling(5).mean()

        # RSI
        g[f'rsi_{rsi_window}'] = _rsi(g['close'], window=rsi_window)

        # MACD
        # moving average convergence/divergence, is a trading indicator used in technical analysis of securities prices, 
        # designed to reveal changes in the strength, direction, momentum, and duration of a trend in a stock's price. 
        # The MACD indicator is a collection of three time series calculated from historical price data, most often the closing price.
        # Why MACD uses span 12 and 26
        # The 12-day and 26-day EMAs are used in MACD because they effectively capture short-term and medium-term price trends,
        # EMA12 - EMA26 = MACD line/momentum
        # when short-term EMA (12) crosses above long-term EMA (26) -> MACD positive, it signals upward momentum (bullish)
        # when short-term EMA (12) crosses below long-term EMA (26) -> MACD negative, it signals downward momentum (bearish)
        
        
        # EMA stands for Exponential Moving Average, a popular technical indicator that smooths price data to identify trends by giving more weight to recent prices, 
        # making it faster to react than a Simple Moving Average (SMA) and useful for spotting entry/exit points, support/resistance, and overall market direction in stocks, forex, and commodities
        # EMA: smoother than price, faster than SMA, captures momentum
        
        ema12 = g['close'].ewm(span=12, adjust=False).mean()
        ema26 = g['close'].ewm(span=26, adjust=False).mean()
        
        # ewm = Exponentially Weighted Moving, which creates a moving average where:
        # recent values matter more, old values matter less, weights decay exponentially, span controls
        
        g['macd'] = ema12 - ema26
        g['macd_signal'] = g['macd'].ewm(span=9, adjust=False).mean()

        # Calendar features
        g['dayofweek'] = g[time_col].dt.dayofweek
        g['month'] = g[time_col].dt.month
        g['day'] = g[time_col].dt.day

        return g

    out = df.groupby(symbol_col, group_keys=False).apply(_features)
    out = out.reset_index(drop=True)
    return out


if __name__ == '__main__':
    base = os.getcwd()
    sample = os.path.join(base, 'data', 'VNM_2022.csv')
    if not os.path.exists(sample):
        raise SystemExit(f"Sample CSV not found at {sample}")
    src_df = pd.read_csv(sample)
    feats = build_features(src_df)
    out_path = os.path.join(base, 'data', 'features_VNM_2022.csv')
    feats.to_csv(out_path, index=False)
    print(f"Wrote features to {out_path}")

# how can we incoporate label generation into feature building?
# we can import generate_labels function from label_generator/generate_label.py and call it within build_features function
# however, it is better to keep feature building and label generation separate for modularity
# users may want to use features without labels or vice versa
# thus, we keep them as separate functions and let users call them as needed in their pipeline  





























