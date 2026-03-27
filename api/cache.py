import logging
import time

from portfolio_optimization.config import Config
from portfolio_optimization.data.collector import DataCollector

logger = logging.getLogger(__name__)

config = Config()

MASTER_CACHE = {
    'data': None,
    'timestamp': 0,
    'ttl': 3600
}
RESULT_CACHE = {}
CACHE_TTL = 300


def get_master_data():
    """Load all data once and cache for 1 hour."""
    current_time = time.time()

    if MASTER_CACHE['data'] is None or (current_time - MASTER_CACHE['timestamp']) > MASTER_CACHE['ttl']:
        logger.info("Loading master dataset (all assets)...")
        collector = DataCollector()
        price_data = collector.fetch_stock_data(config.all_tickers, period=config.data_period)
        returns_data = collector.calculate_returns()

        MASTER_CACHE['data'] = {
            'prices': price_data,
            'returns': returns_data,
            'collector': collector
        }
        MASTER_CACHE['timestamp'] = current_time
        logger.info(f"Master data loaded: {len(returns_data.columns)} assets, {len(returns_data)} days")

    return MASTER_CACHE['data']


async def get_portfolio_data(requested_assets=None):
    """Get data filtered to requested assets from master cache."""
    master = get_master_data()

    if requested_assets is None:
        return master

    available = [a for a in requested_assets if a in master['returns'].columns]

    return {
        'prices': master['prices'][available] if len(available) > 0 else master['prices'],
        'returns': master['returns'][available] if len(available) > 0 else master['returns'],
        'collector': master['collector']
    }


def get_cached_result(cache_key: str):
    if cache_key in RESULT_CACHE:
        entry = RESULT_CACHE[cache_key]
        if time.time() - entry['timestamp'] < CACHE_TTL:
            return entry['data']
    return None


def set_cached_result(cache_key: str, data):
    RESULT_CACHE[cache_key] = {
        'data': data,
        'timestamp': time.time()
    }
