import pandas as pd
import numpy as np # Added numpy import, might be needed from original logic
from config import COMPRESSION_THRESHOLD, ALERT_THRESHOLD, LOOKBACK_PERIODS

def detect_compression_zones(df, lookback, threshold=COMPRESSION_THRESHOLD):
    """Process dataframe to identify compression zones"""
    # Ensure df is a copy to avoid modifying the original DataFrame passed to the function
    df = df.copy() 
    
    # Check if required columns exist and df is not empty
    if df.empty or lookback <= 0:
        print("DataFrame is empty or lookback period is invalid. Cannot detect zones.")
        # Return df with expected columns, possibly empty or with NaNs
        df['rolling_high'] = np.nan
        df['rolling_low'] = np.nan
        df['range'] = np.nan
        df['compression'] = False
        df['compression_start'] = False
        df['compression_end'] = False
        return df

    # Use high_price and low_price if available, otherwise calculate from price
    if 'high_price' in df.columns and 'low_price' in df.columns and \
       not df['high_price'].isnull().all() and not df['low_price'].isnull().all():
        df['rolling_high'] = df['high_price'].rolling(window=lookback, min_periods=min(lookback, len(df))).max()
        df['rolling_low'] = df['low_price'].rolling(window=lookback, min_periods=min(lookback, len(df))).min()
    elif 'price' in df.columns and not df['price'].isnull().all():
         print("Warning: 'high_price' or 'low_price' not available or all null. Using 'price' for rolling calculations.")
         df['rolling_high'] = df['price'].rolling(window=lookback, min_periods=min(lookback, len(df))).max()
         df['rolling_low'] = df['price'].rolling(window=lookback, min_periods=min(lookback, len(df))).min()
    else:
        print("Error: No usable price columns ('price', 'high_price', 'low_price') found.")
        # Return df with expected columns containing NaNs or False
        df['rolling_high'] = np.nan
        df['rolling_low'] = np.nan
        df['range'] = np.nan
        df['compression'] = False
        df['compression_start'] = False
        df['compression_end'] = False
        return df

    # Calculate range only where rolling_low is not zero or NaN to avoid division errors
    df['range'] = np.where(
        (df['rolling_low'].notna()) & (df['rolling_low'] != 0),
        (df['rolling_high'] - df['rolling_low']) / df['rolling_low'],
        np.nan # Assign NaN where rolling_low is zero or NaN
    )
    
    # Identify compression zones - where price range is below threshold
    # Ensure range is not NaN before comparison
    df['compression'] = (df['range'].notna()) & (df['range'] < threshold)
    
    # Identify transitions (start and end of compression zones)
    # Shift requires at least one previous row, handle edge case for very short dataframes
    if len(df) > 1:
        compression_shifted = df['compression'].shift(1, fill_value=False) # Fill NaN shift with False
        df['compression_start'] = (df['compression'] == True) & (compression_shifted == False)
        df['compression_end'] = (df['compression'] == False) & (compression_shifted == True)
    else:
        df['compression_start'] = df['compression'] # If only one row, it's a start if compressed
        df['compression_end'] = False

    # Calculate relative price (optional, kept from original)
    # if 'price' in df.columns and not df['price'].isnull().all():
    #     df['relative_price'] = df['price'] / df['price'].expanding().max()
    # else:
    #     df['relative_price'] = np.nan
        
    return df

def check_for_breakout(df, lookback, compression_threshold=COMPRESSION_THRESHOLD, 
                     alert_threshold=ALERT_THRESHOLD, timeframe='1d'):
    """Check if price has broken out of compression zone"""
    # Ensure required columns exist and df has enough data
    required_cols = ['price', 'rolling_high', 'rolling_low', 'compression', 'range']
    if df.empty or len(df) < lookback or not all(col in df.columns for col in required_cols):
        # print(f"Not enough data or missing columns for breakout check (Timeframe: {timeframe}, Length: {len(df)}, Lookback: {lookback})")
        return False, None, None # Return default non-breakout tuple

    # Get the most recent data point and the data for the lookback period
    last_row = df.iloc[-1]
    recent_data = df.iloc[-lookback:] # Data over the lookback period ending now

    # Check if the *most recent* data point indicates compression was active
    # Or if compression was active *recently* within the lookback period
    # Using mean range might be misleading if there was a brief spike. Check last point or recent average.
    # Let's check if the last point was marked as compression OR if the average range was low.
    
    # Check if 'compression' column exists and last value is True
    in_compression_now = last_row.get('compression', False) == True
    # Check if average range over lookback period was below threshold (alternative check)
    avg_recent_range = recent_data['range'].mean()
    recently_compressed = pd.notna(avg_recent_range) and avg_recent_range < compression_threshold

    # Consider a breakout check if either condition is met (currently or recently compressed)
    # This handles cases where price breaks out *immediately* after compression ends.
    if in_compression_now or recently_compressed:
        # Calculate compression zone boundaries using the mean of rolling high/low over the lookback period
        zone_high = recent_data['rolling_high'].mean()
        zone_low = recent_data['rolling_low'].mean()
        current_price = last_row['price']
        
        # Check if zone boundaries and current price are valid numbers
        if pd.isna(zone_high) or pd.isna(zone_low) or pd.isna(current_price):
            # print(f"Cannot check breakout due to NaN values (Timeframe: {timeframe})")
            return False, None, None # Cannot determine breakout

        # Check for breakout
        upper_breakout_level = zone_high * (1 + alert_threshold)
        lower_breakout_level = zone_low * (1 - alert_threshold)
        
        upper_breakout = current_price > upper_breakout_level
        lower_breakout = current_price < lower_breakout_level
        
        if upper_breakout:
            msg = f"UPPER BREAKOUT on {timeframe}! Price: ${current_price:,.2f} broke above recent high avg ${zone_high:,.2f} (Threshold: ${upper_breakout_level:,.2f})"
            return True, "upper", msg
        elif lower_breakout:
            msg = f"LOWER BREAKOUT on {timeframe}! Price: ${current_price:,.2f} broke below recent low avg ${zone_low:,.2f} (Threshold: ${lower_breakout_level:,.2f})"
            return True, "lower", msg
        # else:
            # print(f"Currently/Recently compressed on {timeframe}, but no breakout. Price: ${current_price:.2f}, Zone: ${zone_low:.2f}-${zone_high:.2f}")

    # If not recently compressed or no breakout occurred
    return False, None, None
