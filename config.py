import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API credentials
MEXC_API_KEY = os.getenv('MEXC_API_KEY')
MEXC_SECRET_KEY = os.getenv('MEXC_SECRET_KEY')

# Application settings
SYMBOL = 'BTCUSDT'
REFRESH_INTERVAL = 60  # seconds

# Analysis parameters
LOOKBACK_PERIODS = {
    '1h': 24,    # 24 hours
    '4h': 48,    # 8 days (48 4-hour periods)
    '1d': 14,    # 14 days
    '1w': 8      # 8 weeks
}
TIMEFRAMES = list(LOOKBACK_PERIODS.keys())
DEFAULT_TIMEFRAME = '1d'
COMPRESSION_THRESHOLD = 0.02
ALERT_THRESHOLD = 0.03