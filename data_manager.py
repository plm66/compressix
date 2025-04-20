import threading
import time
from datetime import datetime, timedelta # Added timedelta import
import pandas as pd

from config import TIMEFRAMES, LOOKBACK_PERIODS, REFRESH_INTERVAL
from api_client import get_current_price, get_historical_data
from analysis import detect_compression_zones, check_for_breakout
from utils import play_alert_sound

# Map our internal timeframe names to MEXC API interval strings
# Reference: https://mexcdevelop.github.io/apidocs/spot_v3_en/#kline-candlestick-data
MEXC_INTERVAL_MAP = {
    '1m': '1m',
    '5m': '5m',
    '15m': '15m',
    '30m': '30m',
    '1h': '60m', # Corrected from '1h'
    '4h': '4h',
    '1d': '1d',
    '1w': '1w',
    '1M': '1M' # Note: MEXC uses '1M' for month
}

# Global data store
latest_data = {
    'current_price': None,
    'last_updated': None,
    'alert_active': False,
    'alert_message': '',
    'alert_time': None,
    'timeframe_data': {}
}

def refresh_all_data():
    """Update data for all timeframes and check for alerts"""
    global latest_data # Ensure we modify the global dictionary
    
    alert_fired_this_cycle = False
    combined_alerts = []
    
    try:
        # Fetch current price once per cycle
        current_price = get_current_price()
        if current_price is not None:
            latest_data['current_price'] = current_price
            print(f"Updated current price: {current_price}")
        else:
            print("Failed to fetch current price for this cycle.")
            # Optionally decide if you want to proceed without a current price
            # For now, we'll continue to fetch historical data

        # Process each timeframe
        for timeframe in TIMEFRAMES:
            lookback = LOOKBACK_PERIODS[timeframe]
            # Ensure we fetch enough data for lookback + some buffer
            limit = max(100, lookback * 3)
            
            # Get the correct API interval string
            api_interval = MEXC_INTERVAL_MAP.get(timeframe)
            if not api_interval:
                print(f"Warning: Timeframe '{timeframe}' not found in MEXC_INTERVAL_MAP. Skipping.")
                continue

            print(f"Fetching historical data for timeframe: {timeframe} (API Interval: {api_interval}, limit: {limit})")
            # Pass the correct API interval to the function
            df = get_historical_data(timeframe=api_interval, limit=limit) 
            
            if df.empty:
                print(f"No historical data received for timeframe: {timeframe}. Skipping analysis.")
                latest_data['timeframe_data'][timeframe] = {
                    'df': pd.DataFrame(), # Store empty df
                    'breakout': False,
                    'direction': None,
                    'alert_message': None
                }
                continue # Skip to next timeframe

            # Add current price to the latest point if available and recent enough
            if current_price is not None and not df.empty:
                 latest_kline_time = df.index[-1]
                 # Add if current price is newer than the last kline timestamp
                 # (Using a small buffer like 1 minute for safety)
                 if datetime.now(latest_kline_time.tz) - latest_kline_time > timedelta(minutes=1):
                     new_row = pd.DataFrame({
                         'price': [current_price],
                         'high_price': [current_price], # Use current price for high/low
                         'low_price': [current_price]
                     }, index=[datetime.now(latest_kline_time.tz)]) # Match timezone if exists
                     df = pd.concat([df, new_row])


            print(f"Analyzing data for timeframe: {timeframe}")
            # Process data to detect compression zones
            df_processed = detect_compression_zones(df, lookback=lookback)
            
            # Check for breakouts
            breakout, direction, alert_message = check_for_breakout(
                df_processed, lookback=lookback, timeframe=timeframe)
            
            # Store the processed data
            latest_data['timeframe_data'][timeframe] = {
                'df': df_processed,
                'breakout': breakout,
                'direction': direction,
                'alert_message': alert_message
            }
            
            # If breakout detected, collect alert
            if breakout:
                alert_fired_this_cycle = True
                combined_alerts.append(alert_message)
                print(f"Breakout detected on {timeframe}: {alert_message}")

        # Update global alert status only if a new alert was fired in this cycle
        if alert_fired_this_cycle:
            # Only trigger sound/update message if the alert wasn't already active
            # or if the message is different (e.g., different timeframe breakout)
            new_alert_message = " | ".join(combined_alerts)
            if not latest_data.get('alert_active', False) or latest_data.get('alert_message') != new_alert_message:
                latest_data['alert_active'] = True
                latest_data['alert_message'] = new_alert_message
                latest_data['alert_time'] = datetime.now()
                print(f"*** ALERT ACTIVATED: {latest_data['alert_message']} ***")
                # Play alert sound in a separate thread
                threading.Thread(target=play_alert_sound, daemon=True).start()
            else:
                 print("Alert condition persists, but alert already active with the same message.")
        # If no alert fired this cycle, we don't deactivate existing alerts here.
        # Deactivation should happen via user action (e.g., 'Clear Alerts' button).

        # Update last updated time after all processing
        latest_data['last_updated'] = datetime.now()

    except Exception as e:
        print(f"Error during data refresh cycle: {str(e)}")
        # Optionally reset parts of latest_data or handle specific errors

    # No return needed as we modify the global latest_data directly

def background_refresh():
    """Thread function to periodically refresh data"""
    while True:
        try:
            refresh_all_data()
            print(f"Data refreshed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e:
            print(f"Error in background refresh: {str(e)}")
        
        time.sleep(REFRESH_INTERVAL)

def start_background_refresh():
    """Start the background data refresh thread"""
    refresh_thread = threading.Thread(target=background_refresh, daemon=True)
    refresh_thread.start()
    return refresh_thread
