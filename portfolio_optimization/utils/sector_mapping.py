"""
Sector mapping utility for portfolio analysis and visualization
"""

SECTOR_MAPPING = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Technology',
    'NVDA': 'Technology', 'META': 'Technology', 'TSLA': 'Technology', 'NFLX': 'Technology',
    'ADBE': 'Technology', 'CRM': 'Technology', 'ORCL': 'Technology', 'INTC': 'Technology',
    'AMD': 'Technology', 'QCOM': 'Technology', 'AVGO': 'Technology', 'TXN': 'Technology',
    'CSCO': 'Technology', 'NOW': 'Technology', 'INTU': 'Technology', 'AMAT': 'Technology',
    'SNOW': 'Technology', 'PLTR': 'Technology', 'CRWD': 'Technology', 'ZS': 'Technology',
    'DDOG': 'Technology', 'NET': 'Technology', 'OKTA': 'Technology', 'TWLO': 'Technology',
    'SHOP': 'Technology', 'PYPL': 'Technology', 'ZM': 'Technology',
    'DOCU': 'Technology', 'MDB': 'Technology',

    # Financials
    'JPM': 'Financials', 'V': 'Financials', 'MA': 'Financials', 'BAC': 'Financials',
    'WFC': 'Financials', 'GS': 'Financials', 'MS': 'Financials', 'C': 'Financials',
    'AXP': 'Financials', 'BLK': 'Financials', 'SPGI': 'Financials', 'CME': 'Financials',
    'ICE': 'Financials', 'MCO': 'Financials', 'MSCI': 'Financials', 'COF': 'Financials',
    'USB': 'Financials', 'PNC': 'Financials', 'TFC': 'Financials', 'SCHW': 'Financials',

    # Healthcare
    'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare', 'ABBV': 'Healthcare',
    'TMO': 'Healthcare', 'ABT': 'Healthcare', 'LLY': 'Healthcare', 'BMY': 'Healthcare',
    'MRK': 'Healthcare', 'AMGN': 'Healthcare', 'GILD': 'Healthcare', 'REGN': 'Healthcare',
    'VRTX': 'Healthcare', 'BIIB': 'Healthcare', 'ISRG': 'Healthcare',
    'DHR': 'Healthcare', 'BSX': 'Healthcare', 'MDT': 'Healthcare', 'SYK': 'Healthcare',

    # Consumer Discretionary
    'HD': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary', 'SBUX': 'Consumer Discretionary',
    'MCD': 'Consumer Discretionary', 'LOW': 'Consumer Discretionary', 'TJX': 'Consumer Discretionary',
    'BKNG': 'Consumer Discretionary', 'DIS': 'Consumer Discretionary', 'CMG': 'Consumer Discretionary',
    'LULU': 'Consumer Discretionary', 'YUM': 'Consumer Discretionary', 'MAR': 'Consumer Discretionary',
    'HLT': 'Consumer Discretionary', 'MGM': 'Consumer Discretionary', 'LVS': 'Consumer Discretionary',
    'WYNN': 'Consumer Discretionary',

    # Consumer Staples
    'PG': 'Consumer Staples', 'COST': 'Consumer Staples', 'WMT': 'Consumer Staples',
    'KO': 'Consumer Staples', 'PEP': 'Consumer Staples',
    'CVS': 'Consumer Staples', 'TGT': 'Consumer Staples', 'KR': 'Consumer Staples',
    'MNST': 'Consumer Staples', 'CL': 'Consumer Staples', 'GIS': 'Consumer Staples',
    'HSY': 'Consumer Staples', 'SJM': 'Consumer Staples',
    'CAG': 'Consumer Staples', 'CPB': 'Consumer Staples', 'MKC': 'Consumer Staples',
    'CHD': 'Consumer Staples', 'CLX': 'Consumer Staples',

    # Industrials
    'BA': 'Industrials', 'CAT': 'Industrials', 'GE': 'Industrials', 'HON': 'Industrials',
    'UNP': 'Industrials', 'UPS': 'Industrials', 'FDX': 'Industrials', 'RTX': 'Industrials',
    'LMT': 'Industrials', 'NOC': 'Industrials', 'GD': 'Industrials', 'MMM': 'Industrials',
    'DE': 'Industrials', 'EMR': 'Industrials', 'ETN': 'Industrials', 'ITW': 'Industrials',
    'PH': 'Industrials', 'CMI': 'Industrials', 'ROK': 'Industrials', 'DOV': 'Industrials',

    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'EOG': 'Energy', 'SLB': 'Energy',
    'KMI': 'Energy', 'OKE': 'Energy', 'WMB': 'Energy', 'VLO': 'Energy',
    'PSX': 'Energy', 'MPC': 'Energy', 'DVN': 'Energy', 'FANG': 'Energy',
    'APA': 'Energy', 'OXY': 'Energy', 'HAL': 'Energy', 'BKR': 'Energy',

    # Materials
    'LIN': 'Materials', 'APD': 'Materials', 'FCX': 'Materials', 'NEM': 'Materials',
    'GOLD': 'Materials', 'AA': 'Materials', 'CLF': 'Materials',
    'NUE': 'Materials', 'STLD': 'Materials', 'VMC': 'Materials', 'MLM': 'Materials',
    'EMN': 'Materials', 'DD': 'Materials', 'DOW': 'Materials', 'LYB': 'Materials',
    'CF': 'Materials', 'MOS': 'Materials', 'FMC': 'Materials', 'ALB': 'Materials',

    # Utilities
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'D': 'Utilities',
    'EXC': 'Utilities', 'XEL': 'Utilities', 'SRE': 'Utilities', 'AEP': 'Utilities',
    'PCG': 'Utilities', 'ED': 'Utilities', 'ES': 'Utilities', 'FE': 'Utilities',
    'ETR': 'Utilities', 'PPL': 'Utilities', 'AES': 'Utilities', 'NRG': 'Utilities',
    'VST': 'Utilities', 'CEG': 'Utilities', 'AWK': 'Utilities', 'ATO': 'Utilities',

    # Fixed Income
    'TLT': 'Fixed Income', 'IEF': 'Fixed Income', 'SHY': 'Fixed Income', 'TIP': 'Fixed Income',
    'SCHZ': 'Fixed Income', 'GOVT': 'Fixed Income', 'IEI': 'Fixed Income', 'SHV': 'Fixed Income',
    'BIL': 'Fixed Income', 'SCHO': 'Fixed Income', 'LQD': 'Fixed Income', 'HYG': 'Fixed Income',
    'EMB': 'Fixed Income', 'JNK': 'Fixed Income', 'VCIT': 'Fixed Income', 'VCSH': 'Fixed Income',
    'IGSB': 'Fixed Income', 'USIG': 'Fixed Income', 'IGIB': 'Fixed Income', 'SJNK': 'Fixed Income',

    # Commodities
    'GLD': 'Commodities', 'SLV': 'Commodities', 'DBA': 'Commodities', 'USO': 'Commodities',
    'UNG': 'Commodities', 'PDBC': 'Commodities', 'GSG': 'Commodities', 'DJP': 'Commodities',
    'CORN': 'Commodities', 'WEAT': 'Commodities',

    # International
    'VEA': 'International', 'EFA': 'International', 'IEFA': 'International', 'SCHF': 'International',
    'VTEB': 'International', 'IEUR': 'International', 'EWJ': 'International', 'EWG': 'International',
    'EWU': 'International', 'EWC': 'International', 'VWO': 'International', 'IEMG': 'International',
    'EEM': 'International', 'SCHE': 'International', 'FXI': 'International', 'EWZ': 'International',
    'INDA': 'International', 'RSX': 'International', 'EWT': 'International', 'EWY': 'International',

    # REITs
    'VNQ': 'REITs', 'IYR': 'REITs', 'XLRE': 'REITs', 'SCHH': 'REITs', 'RWR': 'REITs',
    'FREL': 'REITs', 'REZ': 'REITs', 'MORT': 'REITs', 'REM': 'REITs', 'USRT': 'REITs'
}

SECTOR_COLORS = {
    'Technology': '#58a6ff',
    'Financials': '#3fb950',
    'Healthcare': '#f85149',
    'Consumer Discretionary': '#ffd700',
    'Consumer Staples': '#8b949e',
    'Industrials': '#6e7681',
    'Energy': '#ff8c00',
    'Materials': '#9d4edd',
    'Utilities': '#06d6a0',
    'Fixed Income': '#2b2d42',
    'Commodities': '#e76f51',
    'International': '#f4a261',
    'REITs': '#264653'
}

def get_sector(symbol: str) -> str:
    """Get sector for a given symbol"""
    return SECTOR_MAPPING.get(symbol, 'Other')

def get_sector_color(sector: str) -> str:
    """Get color for a given sector"""
    return SECTOR_COLORS.get(sector, '#30363d')

def calculate_sector_exposure(weights: dict) -> dict:
    """Calculate sector exposure from portfolio weights"""
    sector_weights = {}

    for symbol, weight in weights.items():
        sector = get_sector(symbol)
        if sector not in sector_weights:
            sector_weights[sector] = 0
        sector_weights[sector] += weight

    return sector_weights

def get_sector_symbols(sector: str) -> list:
    """Get all symbols for a given sector"""
    return [symbol for symbol, sec in SECTOR_MAPPING.items() if sec == sector]
