#!/usr/bin/env python3

"""
Bitcoin Compression Zone Monitor
Main application entry point
"""

import sys
from dash_app import create_app
from data_manager import start_background_refresh, refresh_all_data

def main():
    """Initialize and run the application"""
    print("Starting Bitcoin Compression Zone Monitor...")
    print("Pre-loading data for all timeframes...")
    refresh_all_data()
    
    app = create_app()
    # Start the background refresh thread
    refresh_thread = start_background_refresh()
    
    # Run the web application
    print("Starting web interface...")
    app.run(debug=True, host='0.0.0.0')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)