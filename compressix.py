import dash
from dash import dcc, html, Input, Output, State, callback, callback_context
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import time
import json
import hmac
import hashlib
import os
from urllib.parse import urlencode
from dash.exceptions import PreventUpdate
from dotenv import load_dotenv
import threading
import subprocess
import platform
import flask

# Load environment variables from .env file
load_dotenv()

# MEXC API credentials
MEXC_API_KEY = os.getenv('MEXC_API_KEY')
MEXC_SECRET_KEY = os.getenv('MEXC_SECRET_KEY')

# Initialize the Dash app
server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)

# Define compression detection parameters
LOOKBACK_PERIODS = {
    '1h': 24,    # 24 hours
    '4h': 48,    # 8 days (48 4-hour periods)
    '1d': 14,    # 14 days
    '1w': 8      # 8 weeks
}
TIMEFRAMES = list(LOOKBACK_PERIODS.keys())
COMPRESSION_THRESHOLD = 0.02  # 2% threshold for compression detection
ALERT_THRESHOLD = 0.03  # 3% move out of compression zone triggers alert
SYMBOL = 'BTCUSDT'  # Trading pair for Bitcoin on MEXC (no underscore)
DEFAULT_TIMEFRAME = '1d'  # Default timeframe to show
REFRESH_INTERVAL = 60  # Refresh interval in seconds

# Store for latest data and alert status
latest_data = {
    'current_price': None,
    'last_updated': None,
    'alert_active': False,
    'alert_message': '',
    'alert_time': None,
    'timeframe_data': {}
}

# Function to play alert sound (in a separate thread to avoid blocking)
def play_alert_sound():
    try:
        # Detect operating system
        os_name = platform.system()
        
        if os_name == "Windows":
            # On Windows, we would use winsound, but we'll use print instead for portability
            print("\a")  # ASCII bell character
        elif os_name == "Darwin":  # macOS
            # Use macOS 'afplay' to play the system alert sound
            subprocess.call(["afplay", "/System/Library/Sounds/Ping.aiff"])
        else:  # Linux or other Unix
            # Use the console bell on Unix systems
            print("\a")
            
        # Always print a message to the console
        print("*** ALERT: Price breakout detected! ***")
    except Exception as e:
        # Fallback in case of any error
        print("\a")  # ASCII bell character
        print(f"*** ALERT: Price breakout detected! (Sound error: {e}) ***")

# Function to fetch Bitcoin price data from MEXC
def get_bitcoin_data(timeframe='1d', limit=100):
    try:
        # First, let's get the latest price using the ticker endpoint
        ticker_url = f'https://api.mexc.com/api/v3/ticker/price?symbol={SYMBOL}'
        ticker_response = requests.get(ticker_url)
        
        current_price = None
        if ticker_response.status_code != 200:
            print(f"Error fetching current price: {ticker_response.text}")
        else:
            current_price = float(ticker_response.json()['price'])
            print(f"Current BTC price: ${current_price:.2f}")
            latest_data['current_price'] = current_price
            latest_data['last_updated'] = datetime.now()
        
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
            print(f"Error from MEXC API: {response.text}")
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
        
        # If we got the current price from ticker, add it as the most recent point
        if current_price is not None and len(df) > 0:
            latest_time = df.index[-1]
            # Only add if it's significantly newer than the last kline
            if datetime.now() - latest_time > timedelta(minutes=30):
                new_row = pd.DataFrame({
                    'price': [current_price],
                    'high_price': [current_price],
                    'low_price': [current_price]
                }, index=[datetime.now()])
                df = pd.concat([df, new_row])
        
        return df
    
    except Exception as e:
        print(f"Error fetching data from MEXC: {str(e)}")
        
        # Fallback: If we can't get MEXC data, create a sample dataset
        dates = pd.date_range(end=datetime.now(), periods=100)
        prices = np.linspace(80000, 85000, 100) + np.random.normal(0, 1000, 100)
        df = pd.DataFrame({
            'price': prices,
            'high_price': prices + np.random.normal(0, 200, 100),
            'low_price': prices - np.random.normal(0, 200, 100)
        }, index=dates)
        print("Using fallback sample data with current BTC price range")
        return df

# Function to detect compression zones
def detect_compression_zones(df, lookback, threshold=COMPRESSION_THRESHOLD):
    df = df.copy()
    
    # Use high_price and low_price if available, otherwise calculate from price
    if 'high_price' in df.columns and 'low_price' in df.columns:
        df['rolling_high'] = df['high_price'].rolling(lookback).max()
        df['rolling_low'] = df['low_price'].rolling(lookback).min()
    else:
        df['rolling_high'] = df['price'].rolling(lookback).max()
        df['rolling_low'] = df['price'].rolling(lookback).min()
    
    df['range'] = (df['rolling_high'] - df['rolling_low']) / df['rolling_low']
    
    # Identify compression zones - where price range is below threshold
    df['compression'] = df['range'] < threshold
    
    # Identify transitions (start and end of compression zones)
    df['compression_start'] = (df['compression'] == True) & (df['compression'].shift(1) == False)
    df['compression_end'] = (df['compression'] == False) & (df['compression'].shift(1) == True)
    
    # Calculate relative price (current price / all-time high in the dataset)
    df['relative_price'] = df['price'] / df['price'].expanding().max()
    
    return df

# Function to check for breakouts from compression zones
def check_for_breakout(df, lookback, compression_threshold=COMPRESSION_THRESHOLD, 
                      alert_threshold=ALERT_THRESHOLD, timeframe='1d'):
    if len(df) < lookback:
        return False, None, None
    
    # Get most recent data based on lookback period
    recent_data = df.iloc[-lookback:]
    
    # Check if there's evidence of compression
    compression_detected = recent_data['range'].mean() < compression_threshold
    
    if compression_detected:
        # Calculate compression zone boundaries
        zone_high = recent_data['rolling_high'].mean()
        zone_low = recent_data['rolling_low'].mean()
        current_price = df['price'].iloc[-1]
        
        # Check for breakout
        upper_breakout = current_price > zone_high * (1 + alert_threshold)
        lower_breakout = current_price < zone_low * (1 - alert_threshold)
        
        if upper_breakout:
            msg = f"UPPER BREAKOUT on {timeframe}! Price: ${current_price:.2f} broke above ${zone_high:.2f}"
            return True, "upper", msg
        elif lower_breakout:
            msg = f"LOWER BREAKOUT on {timeframe}! Price: ${current_price:.2f} broke below ${zone_low:.2f}"
            return True, "lower", msg
        
    return False, None, None

# Function to refresh data for all timeframes
def refresh_all_data():
    global latest_data
    
    alert_fired = False
    combined_alerts = []
    
    for timeframe in TIMEFRAMES:
        lookback = LOOKBACK_PERIODS[timeframe]
        limit = max(100, lookback * 3)  # Ensure we get enough data
        
        # Get data for this timeframe
        df = get_bitcoin_data(timeframe=timeframe, limit=limit)
        
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
        if breakout and not alert_fired:
            alert_fired = True
            combined_alerts.append(alert_message)
    
    # Update alert status
    if alert_fired:
        latest_data['alert_active'] = True
        latest_data['alert_message'] = " | ".join(combined_alerts)
        latest_data['alert_time'] = datetime.now()
        
        # Play alert sound in a separate thread
        threading.Thread(target=play_alert_sound).start()
    
    return latest_data

# Background data refresh thread
def background_refresh():
    while True:
        try:
            refresh_all_data()
            print(f"Data refreshed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e:
            print(f"Error in background refresh: {str(e)}")
        
        # Sleep until next refresh
        time.sleep(REFRESH_INTERVAL)

# Start background refresh in a separate thread
refresh_thread = threading.Thread(target=background_refresh, daemon=True)
refresh_thread.start()

# App layout
app.layout = html.Div([
    html.H1("Bitcoin Compression Zone Monitor (MEXC)", style={'textAlign': 'center'}),
    
    # Current price display
    html.Div([
        html.H2("Current BTC Price:", style={'display': 'inline-block', 'marginRight': '10px'}),
        html.H2(id='current-price', style={'display': 'inline-block', 'color': 'green', 'fontWeight': 'bold'}),
        html.Div(id='last-update-time', style={'fontSize': '0.8em', 'color': 'gray'}),
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),
    
    # Alert box
    html.Div(id='alert-box', style={
        'backgroundColor': 'rgba(255, 200, 200, 0.5)',
        'padding': '10px',
        'borderRadius': '5px',
        'marginBottom': '20px',
        'display': 'none'  # Initially hidden
    }),
    
    # Control panel
    html.Div([
        html.Div([
            html.Label("Select Timeframe:"),
            dcc.Dropdown(
                id='timeframe-dropdown',
                options=[{'label': tf, 'value': tf} for tf in TIMEFRAMES],
                value=DEFAULT_TIMEFRAME,
                clearable=False
            ),
        ], style={'width': '20%', 'display': 'inline-block'}),
        
        html.Div([
            html.Button('Refresh Data', id='refresh-button', n_clicks=0, 
                       style={'marginRight': '10px'}),
            html.Button('Clear Alerts', id='clear-alerts-button', n_clicks=0)
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'bottom'})
    ], style={'marginBottom': '20px'}),
    
    # Main chart
    dcc.Graph(id='bitcoin-chart', style={'height': '600px'}),
    
    # Status indicators for all timeframes
    html.Div([
        html.H3("Compression Status by Timeframe:"),
        html.Div(id='timeframe-status', style={'marginBottom': '20px'})
    ]),
    
    # Hidden divs for storing state
    dcc.Interval(
        id='interval-component',
        interval=5 * 1000,  # update every 5 seconds
        n_intervals=0
    ),
    dcc.Store(id='app-state')
])

# Callback for current price display
@callback(
    Output('current-price', 'children'),
    Output('current-price', 'style'),
    Output('last-update-time', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_price_display(n_intervals):
    if latest_data['current_price'] is None:
        return "Loading...", {'display': 'inline-block', 'color': 'gray'}, ""
    
    # Format price with commas for thousands
    price_str = f"${latest_data['current_price']:,.2f}"
    
    # Determine color based on price trend (could be enhanced with true trend detection)
    price_style = {'display': 'inline-block', 'fontWeight': 'bold', 'fontSize': '1.8em'}
    
    # Use green by default
    price_style['color'] = 'green'
    
    # Format last updated time
    if latest_data['last_updated']:
        last_updated = f"Last updated: {latest_data['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}"
    else:
        last_updated = ""
    
    return price_str, price_style, last_updated

# Callback for alert box
@callback(
    Output('alert-box', 'children'),
    Output('alert-box', 'style'),
    Input('interval-component', 'n_intervals'),
    Input('clear-alerts-button', 'n_clicks')
)
def update_alert_box(n_intervals, clear_clicks):
    ctx = callback_context
    
    # If clear button was clicked, reset alerts
    if ctx.triggered and 'clear-alerts-button' in ctx.triggered[0]['prop_id']:
        latest_data['alert_active'] = False
        latest_data['alert_message'] = ''
        return "", {'display': 'none'}
    
    if latest_data['alert_active']:
        alert_time = latest_data['alert_time'].strftime('%Y-%m-%d %H:%M:%S') if latest_data['alert_time'] else ""
        alert_content = html.Div([
            html.H3("⚠️ ALERT ⚠️", style={'color': 'red', 'textAlign': 'center'}),
            html.P(latest_data['alert_message'], style={'fontWeight': 'bold'}),
            html.P(f"Detected at: {alert_time}", style={'fontSize': '0.8em'})
        ])
        
        alert_style = {
            'backgroundColor': 'rgba(255, 200, 200, 0.8)',
            'padding': '10px',
            'borderRadius': '5px',
            'marginBottom': '20px',
            'border': '2px solid red',
            'display': 'block'
        }
        
        return alert_content, alert_style
    else:
        return "", {'display': 'none'}

# Callback for timeframe status indicators
@callback(
    Output('timeframe-status', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_timeframe_status(n_intervals):
    status_items = []
    
    for tf in TIMEFRAMES:
        if tf not in latest_data['timeframe_data']:
            status = html.Span("No data", style={'color': 'gray'})
        else:
            tf_data = latest_data['timeframe_data'][tf]
            
            if tf_data['breakout']:
                if tf_data['direction'] == 'upper':
                    status = html.Span("BREAKOUT (UP) ↑", 
                                      style={'color': 'green', 'fontWeight': 'bold'})
                else:
                    status = html.Span("BREAKOUT (DOWN) ↓", 
                                      style={'color': 'red', 'fontWeight': 'bold'})
            else:
                # Check if in compression zone
                if len(tf_data['df']) > 0 and tf_data['df']['compression'].iloc[-1]:
                    status = html.Span("Compression Zone", 
                                      style={'color': 'blue', 'fontStyle': 'italic'})
                else:
                    status = html.Span("Normal", style={'color': 'black'})
        
        status_items.append(
            html.Div([
                html.Span(f"{tf}: ", style={'fontWeight': 'bold', 'width': '40px', 'display': 'inline-block'}),
                status
            ], style={'margin': '5px 0'})
        )
    
    return status_items

# Callback for button and chart updates
@callback(
    Output('bitcoin-chart', 'figure'),
    Output('app-state', 'data'),
    Input('refresh-button', 'n_clicks'),
    Input('interval-component', 'n_intervals'),
    Input('timeframe-dropdown', 'value'),
    State('app-state', 'data')
)
def update_chart(n_clicks, n_intervals, selected_timeframe, app_state):
    if app_state is None:
        app_state = {'last_refresh': 0}
        
    ctx = callback_context
    
    # Force refresh if button clicked
    if ctx.triggered and 'refresh-button' in ctx.triggered[0]['prop_id']:
        refresh_all_data()
        app_state['last_refresh'] = time.time()
    
    # Ensure the selected timeframe exists in our data
    if selected_timeframe not in latest_data['timeframe_data']:
        for tf in TIMEFRAMES:
            if tf in latest_data['timeframe_data']:
                selected_timeframe = tf
                break
    
    # If we still don't have data, just return an empty chart
    if selected_timeframe not in latest_data['timeframe_data']:
        fig = go.Figure()
        fig.update_layout(title="No data available yet. Please wait or click Refresh.")
        return fig, app_state
    
    # Get the data for the selected timeframe
    tf_data = latest_data['timeframe_data'][selected_timeframe]
    df = tf_data['df']
    lookback = LOOKBACK_PERIODS[selected_timeframe]
    
    # Create chart
    fig = go.Figure()
    
    # Add candlestick chart if we have high/low data
    if 'high_price' in df.columns and 'low_price' in df.columns:
        # Create candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['price'],  # Using close price as open since we don't have true open
            high=df['high_price'],
            low=df['low_price'],
            close=df['price'],
            name="Price",
            increasing_line_color='green',
            decreasing_line_color='red'
        ))
    else:
        # Add regular price line
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['price'],
            mode='lines',
            name='Bitcoin Price',
            line={'color': 'blue'}
        ))
    
    # Highlight compression zones
    compressed_regions = []
    in_compression = False
    start_idx = None
    
    for i, (idx, row) in enumerate(df.iterrows()):
        if row['compression_start']:
            in_compression = True
            start_idx = idx
        elif row['compression_end']:
            if in_compression and start_idx:
                compressed_regions.append((start_idx, idx))
                in_compression = False
                start_idx = None
    
    # If still in compression at the end
    if in_compression and start_idx:
        compressed_regions.append((start_idx, df.index[-1]))
    
    # Add compression zones as shapes
    for start, end in compressed_regions:
        zone_data = df.loc[start:end]
        if not zone_data.empty:
            high_val = zone_data['rolling_high'].max()
            low_val = zone_data['rolling_low'].min()
            
            fig.add_shape(
                type="rect",
                x0=start,
                x1=end,
                y0=low_val,
                y1=high_val,
                fillcolor="rgba(173, 216, 230, 0.4)",  # Light blue
                line=dict(width=0),
                layer="below"
            )
            
            # Add label for compression zone
            fig.add_annotation(
                x=start + (end - start)/2,
                y=high_val,
                text="Compression Zone",
                showarrow=False,
                yshift=10,
                font=dict(size=10, color="blue")
            )
    
    # Add high and low lines for the most recent period
    if len(df) >= lookback:
        recent_df = df.iloc[-lookback:]
        is_compressed = recent_df['compression'].any()
        
        if is_compressed:
            recent_high = recent_df['rolling_high'].mean()
            recent_low = recent_df['rolling_low'].mean()
            
            # Add high line
            fig.add_trace(go.Scatter(
                x=[recent_df.index[0], df.index[-1]],
                y=[recent_high, recent_high],
                mode='lines',
                name='Compression High',
                line={'color': 'red', 'dash': 'dash', 'width': 2}
            ))
            
            # Add low line
            fig.add_trace(go.Scatter(
                x=[recent_df.index[0], df.index[-1]],
                y=[recent_low, recent_low],
                mode='lines',
                name='Compression Low',
                line={'color': 'green', 'dash': 'dash', 'width': 2}
            ))
            
            # Add alert thresholds
            fig.add_trace(go.Scatter(
                x=[recent_df.index[0], df.index[-1]],
                y=[recent_high * (1 + ALERT_THRESHOLD), recent_high * (1 + ALERT_THRESHOLD)],
                mode='lines',
                name='Upper Breakout Level',
                line={'color': 'red', 'dash': 'dot'}
            ))
            
            fig.add_trace(go.Scatter(
                x=[recent_df.index[0], df.index[-1]],
                y=[recent_low * (1 - ALERT_THRESHOLD), recent_low * (1 - ALERT_THRESHOLD)],
                mode='lines',
                name='Lower Breakout Level',
                line={'color': 'green', 'dash': 'dot'}
            ))
    
    # Highlight the current price with a marker
    if latest_data['current_price'] is not None:
        fig.add_trace(go.Scatter(
            x=[df.index[-1]],
            y=[latest_data['current_price']],
            mode='markers',
            name='Current Price',
            marker=dict(
                color='gold',
                size=12,
                line=dict(
                    color='black',
                    width=2
                )
            )
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Bitcoin Price with Compression Zones ({selected_timeframe} Timeframe)',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        height=600,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig, app_state

# Run the app
if __name__ == '__main__':
    # Pre-load data for all timeframes
    print("Pre-loading data for all timeframes...")
    refresh_all_data()
    print("Starting application...")
    app.run(debug=True, host='0.0.0.0')