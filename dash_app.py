import dash
from dash import dcc, html, Input, Output, State, callback, callback_context
import plotly.graph_objects as go
from datetime import datetime, timedelta # Added timedelta
import time
import flask
import pandas as pd # Added pandas

from config import TIMEFRAMES, DEFAULT_TIMEFRAME, LOOKBACK_PERIODS, REFRESH_INTERVAL, ALERT_THRESHOLD # Added ALERT_THRESHOLD
from data_manager import latest_data, refresh_all_data

def create_app():
    """Create and configure the Dash application"""
    server = flask.Flask(__name__)
    # Consider adding suppress_callback_exceptions=True if needed later, but start without it.
    app = dash.Dash(__name__, server=server) 
    
    # App layout definition - Rebuilt from compressix.py
    app.layout = html.Div([
        html.H1("Bitcoin Compression Zone Monitor (MEXC)", style={'textAlign': 'center'}),
        
        # Current price display
        html.Div([
            html.H2("Current BTC Price:", style={'display': 'inline-block', 'marginRight': '10px'}),
            # Style will be updated by callback
            html.H2(id='current-price', style={'display': 'inline-block', 'color': 'gray', 'fontWeight': 'bold'}), 
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
            ], style={'width': '20%', 'display': 'inline-block', 'paddingRight': '10px'}),
            
            html.Div([
                html.Button('Refresh Data', id='refresh-button', n_clicks=0, 
                           style={'marginRight': '10px'}),
                html.Button('Clear Alerts', id='clear-alerts-button', n_clicks=0)
            ], style={'display': 'inline-block', 'verticalAlign': 'bottom'})
        ], style={'marginBottom': '20px', 'paddingLeft': '10px'}),
        
        # Main chart
        dcc.Graph(id='bitcoin-chart', style={'height': '600px'}),
        
        # Status indicators for all timeframes
        html.Div([
            html.H3("Compression Status by Timeframe:"),
            html.Div(id='timeframe-status', style={'marginBottom': '20px', 'paddingLeft': '10px'})
        ]),
        
        # Interval component for periodic updates (already present, kept)
        dcc.Interval(
            id='interval-component',
            # Use REFRESH_INTERVAL from config for consistency? Or keep faster UI update?
            # Keeping 5s for faster UI reaction for now.
            interval=5 * 1000,  
            n_intervals=0
        ),
        # Store for application state (e.g., last refresh time)
        dcc.Store(id='app-state') 
    ])
    
    # Define callbacks (will be added/updated in the next step)
    
    # Callback for current price display (full version)
    @callback(
        Output('current-price', 'children'),
        Output('current-price', 'style'),
        Output('last-update-time', 'children'),
        Input('interval-component', 'n_intervals')
    )
    def update_price_display(n_intervals):
        if latest_data['current_price'] is None:
            # Keep initial style if loading
            initial_style = {'display': 'inline-block', 'color': 'gray', 'fontWeight': 'bold'}
            return "Loading...", initial_style, ""
        
        # Format price with commas for thousands
        price_str = f"${latest_data['current_price']:,.2f}"
        
        # Determine color based on price trend (simple version: green for now)
        # TODO: Enhance with actual trend detection if needed
        price_style = {'display': 'inline-block', 'fontWeight': 'bold', 'fontSize': '1.8em', 'color': 'green'}
        
        # Format last updated time
        if latest_data.get('last_updated'):
            last_updated_str = f"Last updated: {latest_data['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}"
        else:
            last_updated_str = "Last updated: Never" # Or handle differently
        
        return price_str, price_style, last_updated_str

    # Callback for alert box
    @callback(
        Output('alert-box', 'children'),
        Output('alert-box', 'style'),
        Input('interval-component', 'n_intervals'),
        Input('clear-alerts-button', 'n_clicks')
    )
    def update_alert_box(n_intervals, clear_clicks):
        ctx = callback_context
        triggered_id = ctx.triggered_id
        
        # If clear button was clicked, reset alerts in latest_data
        if triggered_id == 'clear-alerts-button':
            print("Clear alerts button clicked.")
            latest_data['alert_active'] = False
            latest_data['alert_message'] = ''
            latest_data['alert_time'] = None
            return "", {'display': 'none'} # Return empty content and hide the box
        
        # Check the alert status from latest_data
        if latest_data.get('alert_active', False):
            alert_time_str = latest_data['alert_time'].strftime('%Y-%m-%d %H:%M:%S') if latest_data.get('alert_time') else "N/A"
            alert_content = html.Div([
                html.H3("⚠️ ALERT ⚠️", style={'color': 'red', 'textAlign': 'center'}),
                html.P(latest_data.get('alert_message', 'Unknown alert'), style={'fontWeight': 'bold'}),
                html.P(f"Detected at: {alert_time_str}", style={'fontSize': '0.8em'})
            ])
            
            alert_style = {
                'backgroundColor': 'rgba(255, 200, 200, 0.8)', # More opaque
                'padding': '10px',
                'borderRadius': '5px',
                'marginBottom': '20px',
                'border': '2px solid red',
                'display': 'block' # Make it visible
            }
            
            return alert_content, alert_style
        else:
            # If no active alert, hide the box
            return "", {'display': 'none'}

    # Callback for timeframe status indicators
    @callback(
        Output('timeframe-status', 'children'),
        Input('interval-component', 'n_intervals')
    )
    def update_timeframe_status(n_intervals):
        status_items = []
        
        # Check if timeframe_data exists and is a dictionary
        if not isinstance(latest_data.get('timeframe_data'), dict):
             return html.Div("Timeframe data not yet available or in wrong format.")

        for tf in TIMEFRAMES:
            # Check if data for this specific timeframe exists
            if tf not in latest_data['timeframe_data'] or not isinstance(latest_data['timeframe_data'][tf], dict):
                status = html.Span("Loading data...", style={'color': 'gray'})
            else:
                tf_data = latest_data['timeframe_data'][tf]
                df = tf_data.get('df') # Get the DataFrame
                
                # Check if DataFrame exists and is not empty
                if df is None or df.empty:
                     status = html.Span("No data", style={'color': 'gray'})
                elif tf_data.get('breakout'): # Check for breakout flag
                    if tf_data.get('direction') == 'upper':
                        status = html.Span("BREAKOUT (UP) ↑", 
                                          style={'color': 'green', 'fontWeight': 'bold'})
                    elif tf_data.get('direction') == 'lower':
                        status = html.Span("BREAKOUT (DOWN) ↓", 
                                          style={'color': 'red', 'fontWeight': 'bold'})
                    else: # Breakout True but no direction? Fallback
                         status = html.Span("BREAKOUT (?)", 
                                          style={'color': 'orange', 'fontWeight': 'bold'})
                else:
                    # Check if 'compression' column exists and the last value is True
                    if 'compression' in df.columns and not df.empty and df['compression'].iloc[-1]:
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
        Output('app-state', 'data'), # Output to store state
        Input('refresh-button', 'n_clicks'),
        Input('interval-component', 'n_intervals'),
        Input('timeframe-dropdown', 'value'),
        State('app-state', 'data') # Input state
    )
    def update_chart(n_clicks_refresh, n_intervals, selected_timeframe, app_state):
        # Initialize app_state if it's None (first load)
        if app_state is None:
            app_state = {'last_refresh_trigger': 0} # Use a different key to avoid confusion
            
        ctx = callback_context
        triggered_id = ctx.triggered_id
        
        # Force refresh if refresh button clicked
        # Compare n_clicks with a stored value if needed, but simple trigger is often enough
        if triggered_id == 'refresh-button':
            print("Refresh button clicked, forcing data refresh.")
            refresh_all_data()
            app_state['last_refresh_trigger'] = time.time() # Update state
        
        # --- Data Validation ---
        # Ensure the selected timeframe exists in our data store
        if selected_timeframe not in latest_data.get('timeframe_data', {}):
            print(f"Selected timeframe '{selected_timeframe}' not found in data. Trying default or first available.")
            # Try default timeframe first
            if DEFAULT_TIMEFRAME in latest_data.get('timeframe_data', {}):
                 selected_timeframe = DEFAULT_TIMEFRAME
            else:
                 # Find the first available timeframe in the data
                 available_tfs = list(latest_data.get('timeframe_data', {}).keys())
                 if available_tfs:
                     selected_timeframe = available_tfs[0]
                 else:
                     # No data available at all
                     print("No timeframe data available at all.")
                     fig = go.Figure()
                     fig.update_layout(title="No data available yet. Please wait or click Refresh.")
                     return fig, app_state # Return empty fig and current state

        # Get the specific data for the selected timeframe
        tf_data = latest_data['timeframe_data'].get(selected_timeframe)
        
        # Further check if tf_data or its 'df' is missing or empty
        if tf_data is None or not isinstance(tf_data.get('df'), pd.DataFrame) or tf_data['df'].empty:
            print(f"Data for selected timeframe '{selected_timeframe}' is empty or invalid.")
            fig = go.Figure()
            fig.update_layout(title=f"No data available for timeframe '{selected_timeframe}'. Please wait or click Refresh.")
            return fig, app_state # Return empty fig and current state

        df = tf_data['df']
        lookback = LOOKBACK_PERIODS.get(selected_timeframe, 20) # Default lookback if not found
        
        # --- Chart Creation ---
        fig = go.Figure()
        
        # Add candlestick chart if we have high/low data
        if 'high_price' in df.columns and 'low_price' in df.columns and not df[['high_price', 'low_price']].isnull().all().all():
            fig.add_trace(go.Candlestick(
                x=df.index,
                # Use close price for open if actual open isn't available/reliable
                open=df.get('open_price', df['price']), 
                high=df['high_price'],
                low=df['low_price'],
                close=df['price'],
                name="Price",
                increasing_line_color='green',
                decreasing_line_color='red'
            ))
        else:
            # Fallback to regular price line if no high/low
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['price'],
                mode='lines',
                name='Bitcoin Price',
                line={'color': 'blue'}
            ))
        
        # --- Compression Zone Highlighting ---
        # Check if compression columns exist
        if 'compression_start' in df.columns and 'compression_end' in df.columns and \
           'rolling_high' in df.columns and 'rolling_low' in df.columns:
            
            compressed_regions = []
            in_compression = False
            start_idx = None
            
            # Iterate safely, checking for boolean values
            for idx, row in df.iterrows():
                is_start = row.get('compression_start') == True
                is_end = row.get('compression_end') == True

                if is_start:
                    if not in_compression: # Start of a new zone
                         start_idx = idx
                    in_compression = True # Stay in compression
                elif is_end:
                    if in_compression and start_idx is not None:
                        # End of the current zone
                        compressed_regions.append((start_idx, idx))
                    in_compression = False
                    start_idx = None
                # Handle case where compression continues but isn't explicitly marked as start/end
                elif row.get('compression') == True and not in_compression:
                     start_idx = idx # Implicit start
                     in_compression = True
                elif row.get('compression') == False and in_compression:
                     if start_idx is not None: # Implicit end
                          compressed_regions.append((start_idx, idx))
                     in_compression = False
                     start_idx = None


            # If still in compression at the end of the data
            if in_compression and start_idx is not None:
                compressed_regions.append((start_idx, df.index[-1]))
            
            # Add compression zones as shapes
            for start, end in compressed_regions:
                zone_data = df.loc[start:end]
                if not zone_data.empty and 'rolling_high' in zone_data.columns and 'rolling_low' in zone_data.columns:
                    high_val = zone_data['rolling_high'].max()
                    low_val = zone_data['rolling_low'].min()
                    
                    # Ensure high_val and low_val are valid numbers
                    if pd.notna(high_val) and pd.notna(low_val):
                        fig.add_shape(
                            type="rect",
                            x0=start, x1=end, y0=low_val, y1=high_val,
                            fillcolor="rgba(173, 216, 230, 0.4)", # Light blue
                            line=dict(width=0),
                            layer="below"
                        )
                        # Add label (optional, can clutter)
                        # fig.add_annotation(x=start + (end - start)/2, y=high_val, text="CZ", showarrow=False, yshift=10, font=dict(size=10, color="blue"))
        else:
             print(f"Compression columns missing in DataFrame for timeframe {selected_timeframe}. Skipping zone highlighting.")


        # --- Add High/Low Lines and Breakout Levels for Recent Period ---
        if len(df) >= lookback and 'rolling_high' in df.columns and 'rolling_low' in df.columns:
            recent_df = df.iloc[-lookback:]
            # Check if the *most recent* point is in compression or if there was recent compression
            is_compressed_recently = recent_df['compression'].any() 

            if is_compressed_recently:
                # Use mean of recent rolling high/low for stability
                recent_high = recent_df['rolling_high'].mean()
                recent_low = recent_df['rolling_low'].mean()
                
                if pd.notna(recent_high) and pd.notna(recent_low):
                    # Define start and end points for the lines
                    line_start_x = recent_df.index[0]
                    line_end_x = df.index[-1]

                    # Add Compression High line
                    fig.add_trace(go.Scatter(x=[line_start_x, line_end_x], y=[recent_high, recent_high], mode='lines', name='Compression High', line={'color': 'red', 'dash': 'dash', 'width': 1}))
                    
                    # Add Compression Low line
                    fig.add_trace(go.Scatter(x=[line_start_x, line_end_x], y=[recent_low, recent_low], mode='lines', name='Compression Low', line={'color': 'green', 'dash': 'dash', 'width': 1}))
                    
                    # Add Upper Breakout Level line
                    upper_break_level = recent_high * (1 + ALERT_THRESHOLD)
                    fig.add_trace(go.Scatter(x=[line_start_x, line_end_x], y=[upper_break_level, upper_break_level], mode='lines', name='Upper Breakout Level', line={'color': 'red', 'dash': 'dot', 'width': 1}))
                    
                    # Add Lower Breakout Level line
                    lower_break_level = recent_low * (1 - ALERT_THRESHOLD)
                    fig.add_trace(go.Scatter(x=[line_start_x, line_end_x], y=[lower_break_level, lower_break_level], mode='lines', name='Lower Breakout Level', line={'color': 'green', 'dash': 'dot', 'width': 1}))

        # --- Highlight Current Price ---
        if latest_data.get('current_price') is not None:
            fig.add_trace(go.Scatter(
                x=[df.index[-1]], # Plot at the last timestamp of the historical data
                y=[latest_data['current_price']],
                mode='markers',
                name='Current Price',
                marker=dict(color='gold', size=12, line=dict(color='black', width=2)),
                hovertemplate=f"Current: ${latest_data['current_price']:,.2f}<extra></extra>" # Custom hover
            ))
        
        # --- Final Layout Updates ---
        fig.update_layout(
            title=f'Bitcoin Price with Compression Zones ({selected_timeframe} Timeframe)',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            height=600,
            template='plotly_white', # Cleaner template
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_rangeslider_visible=False # Hide rangeslider for cleaner look
        )
        
        return fig, app_state # Return figure and updated state

    return app
