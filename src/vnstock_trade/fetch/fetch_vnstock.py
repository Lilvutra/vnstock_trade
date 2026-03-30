from vnstock import Vnstock 
from vnstock import Trading 
from vnstock import Quote
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

def fetch_data_v2():
    """
    Get historical stock data for a list of symbols and save to CSV.
    """
    print("Fetching data using Quote API...")
    
    SYMBOLS = ['VNM', 'VHM', 'VCB', 'HPG', 'BID', 'CTG']

    dfs = []
    for sym in SYMBOLS:
        quote = Quote(
            symbol=sym, source='VCI')
        df = quote.history(
            start="2026-03-09",
            end="2026-03-14",
            interval="d"
        )
        df['symbol'] = sym
        dfs.append(df)
 
    market_df = pd.concat(dfs, ignore_index=True).sort_values(by=['time', 'symbol'])
    market_df.to_csv('data/market_data_2026.csv', index=False)
    
def _fetch_data():
    """
        Another function to fetch data which includes price board(bid/ask data, foreign/domestic buy/sell volume), which can be useful for more advanced features and labels
    """
    print("Fetching data using Trading API...")
    
    trading = Trading(source='KBS')
    Symbols = ['VNM', 'VHM', 'VCB', 'HPG', 'BID', 'CTG']
    
    dfs_ = []
    df_ = trading.price_board(['VNM', 'VHM', 'VCB', 'HPG', 'BID', 'CTG'])
    dfs_.append(df_)

    market_df_ = pd.concat(dfs_, ignore_index=True).sort_values(by=['time', 'symbol'])
    market_df_.to_csv('data/market_data_2024_.csv', index=False)



if __name__ == "__main__":
    #fetch_data()
    #_fetch_data()
    #get_data()
    fetch_data_v2()

















