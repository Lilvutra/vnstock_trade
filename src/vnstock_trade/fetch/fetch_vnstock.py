from vnstock import Vnstock 
import pandas as pd 
import os


def fetch_data():
    """
    Get historical stock data for a list of symbols and save to CSV.
    """

    SYMBOLS = ['VNM', 'VHM', 'VCB', 'HPG', 'BID', 'CTG']

    dfs = []
    for sym in SYMBOLS:
        stock = Vnstock().stock(symbol=sym, source='VCI')
        df = stock.quote.history(start='2024-01-01', end='2024-12-31', interval='1D')
        df['symbol'] = sym
        dfs.append(df)
 
    market_df = pd.concat(dfs, ignore_index=True).sort_values(by=['time', 'symbol'])
    market_df.to_csv('data/VNM_2022.csv', index=False)

    # cannot save file to this directory while in a dir that is also a child of main project dir
    # market_df.to_csv('data/market_data_2024.csv', index=False)

if __name__ == "__main__":
    fetch_data()




















