import requests
import time
import pandas as pd
from datetime import datetime, timedelta
from config import MEXC_API_KEY, MEXC_SECRET_KEY, SYMBOL

def get_current_price():
    """Fetch the current BTC price from MEXC"""
    ticker_url = f'https://api.mexc.com/api/v3/ticker/price?symbol={SYMBOL}'
    response = requests.get(ticker_url)
    
    if response.status_code != 200:
        print(f"Error fetching current price: {response.text}")
        return None
    
    return float(response.json()['price'])

def get_historical_data(timeframe='1d', limit=100):
    """Fetch historical price data from MEXC"""
    try:
        # Set up parameters for the klines API request
        params = {
            'symbol': SYMBOL,
            'interval': timeframe,  # Can be 1m, 5m, 15m, 30m, 60m, 4h, 1d, 1w, 1M
            'limit': limit,         # Number of data points
        }
        
        # Make the API request to MEXC for K-line data
        url = 'https://api.mexc.com/api/v3/klines'
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"Error from MEXC API (klines): {response.text}")
            raise Exception(f"MEXC API request failed with status code {response.status_code}")
        
        # Process the data
        data = response.json()
        
        # MEXC kline data format:
        # [0] Open time, [1] Open, [2] High, [3] Low, [4] Close, [5] Volume, etc.
        prices = []
        highs = []
        lows = []
        
        for candle in data:
            timestamp = int(candle[0])  # Open time in milliseconds
            open_price = float(candle[1])
            high_price = float(candle[2])
            low_price = float(candle[3])
            close_price = float(candle[4])  # Close price
            
            prices.append([timestamp, close_price])
            highs.append([timestamp, high_price])
            lows.append([timestamp, low_price])
        
        # Create DataFrame
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['high_price'] = [h[1] for h in highs]
        df['low_price'] = [l[1] for l in lows]
        df = df.set_index('timestamp')
        df = df.sort_index()  # Ensure data is sorted by time
        
        return df

    except Exception as e:
        print(f"Error fetching historical data from MEXC: {str(e)}")
        # Return an empty DataFrame on error to avoid breaking downstream processing
        return pd.DataFrame()
