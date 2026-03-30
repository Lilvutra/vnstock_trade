import os
import numpy as np
import pandas as pd

"""Feature engineering for stock data, including returns, moving averages, volatility, momentum, and technical indicators like RSI and MACD.
Designed for multi-stock daily data with columns: time, open, high, low, close, volume, symbol. Outputs a DataFrame with new feature columns for modeling.

Improvements:
- Minimize features to reduce noise and overfitting, focus on most predictive ones
- Add more technical indicators (e.g., Bollinger Bands, Stochastic Oscillator) for richer signals
- Incorporate fundamental data (e.g., P/E ratio, earnings) for additional perspective beyond price action
- Use feature selection techniques (e.g., mutual information, feature importance) to identify and keep only the most relevant features for the model
- Consider dimensionality reduction (e.g., PCA) if feature space becomes too large, to capture most variance with fewer features

Vietnam market focuses:
Regime classifier (turnover + foreign flow)

Cross-sectional momentum ranking

Volume-adjusted breakout detection

Futures basis as risk filter

Strict walk-forward validation

Keep model simple (e.g., LightGBM or logistic regression)."""

# Relative Strength Index (RSI) calculation for finding overbought or oversold conditions
# Even without volume, price reflects aggregate market behavior:
# If price keeps going up → more aggressive buyers than sellers
# If price keeps going down → selling pressure dominates
# RSI only uses price changes, not actual traded volume, so it approximates buying/selling pressure indirectly. 
# This is a limitation because price alone may not capture conviction. I would consider incorporating volume-based features to improve the signal
def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff() # price change from previous day, positive for gains, negative for losses
    up = delta.clip(lower=0) # only keep gains, set losses to 0
    down = -delta.clip(upper=0) # only keep losses as positive values, set gains to 0
    ma_up = up.rolling(window=window, min_periods=window).mean() # average gain over the window
    ma_down = down.rolling(window=window, min_periods=window).mean() # average loss over the window
    rs = ma_up / ma_down # relative strength, ratio of average gain to average loss
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
            g[f'ma_{w}'] = g['close'].rolling(w).mean() # moving average over w days, 
            g[f'return_{w}'] = g['close'].pct_change(w) # simple return over w days
            g[f'volatility_{w}'] = g['log_return_1'].rolling(w).std() # volatility (std dev of log returns) over w days

        
        # Lagged returns
        for lag in range(1, 6):
            g[f'lag_logret_{lag}'] = g['log_return_1'].shift(lag)
        
        # Momentum and volume features
        g['mom_5'] = g['close'] / g['close'].shift(5) - 1
        g['vol_change'] = g['volume'].pct_change() # percentage change
        g['vol_ma_5'] = g['volume'].rolling(5).mean()

        # RSI - Relative Strength Index, a momentum oscillator that measures the speed and change of price movements, typically used to identify overbought or oversold conditions in a stock, calculated using average gains and losses over a specified period (commonly 14 days), with values ranging from 0 to 100 where above 70 indicates overbought and below 30 indicates oversold conditions
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


def _build_features(df: pd.DataFrame,
                   symbol_col: str = 'symbol',
                   time_col: str = 'time',
                   windows=(5, 10, 21),
                   rsi_window: int = 14) -> pd.DataFrame:
    
    """refined build_features func with more refined features based on Vietnam Market 
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    def features(g_: pd.DataFrame) -> pd.DataFrame:
        
        g_ = g_.sort_values(time_col).copy()
        
        g_['return_1'] = g_['close'].pct_change() # simple daily return
        g_['log_return_1'] = np.log(g_['close']).diff() # log daily return

        # Lagged returns
        for lag in range(1, 6):
            g_[f'lag_logret_{lag}'] = g_['log_return_1'].shift(lag)

        # Momentum and volume features
        g_['mom_5'] = g_['close'] / g_['close'].shift(5) - 1
        g_['vol_change'] = g_['volume'].pct_change() # percentage change
        g_['vol_ma_5'] = g_['volume'].rolling(5).mean()

        # RSI
        g_[f'rsi_{rsi_window}'] = _rsi(g_['close'], window=rsi_window)
        
        # Limit Proximity Feature: Distance to upper limit at close.
        # If stock closes at +6.8%, strong probability of hitting +7% next day with 7% limit, we can capture this by measuring how close the close price is to the upper limit, which is open price * 1.07 for HOSE stocks.
        g_['limit_proximity'] = (g_['close'] - g_['open'] * 1.07) / (g_['open'] * 1.07)
        
        # volume-adjusted proximity 
        g_['vol_adj_limit_proximity'] = g_['limit_proximity'] * g_['volume'] / g_['volume'].rolling(5).mean() # if close is near limit and volume is high relative to recent average, it may indicate strong momentum towards the limit, which could be predictive of hitting the limit next day
         
    
        return g_

    out_ = df.groupby(symbol_col, group_keys=False).apply(features)
    out_ = out_.reset_index(drop=True)
    return out_

if __name__ == '__main__':
    base = os.getcwd()
    sample = os.path.join(base, 'data', 'VNM_2022.csv')
    if not os.path.exists(sample):
        raise SystemExit(f"Sample CSV not found at {sample}")
    src_df = pd.read_csv(sample)
    feats = build_features(src_df)
    out_path = os.path.join(base, 'data', '_features_VNM_2026.csv')
    feats.to_csv(out_path, index=False)
    print(f"Wrote features to {out_path}")

# how can we incoporate label generation into feature building?
# we can import generate_labels function from label_generator/generate_label.py and call it within build_features function
# however, it is better to keep feature building and label generation separate for modularity
# users may want to use features without labels or vice versa
# thus, we keep them as separate functions and let users call them as needed in their pipeline  





























