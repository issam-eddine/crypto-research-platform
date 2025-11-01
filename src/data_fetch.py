import ccxt
import pandas as pd
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

DATA_RAW = Path("data/raw")
DATA_RAW.mkdir(parents=True, exist_ok=True)

exchange = ccxt.binanceus({
    'enableRateLimit': True,
    'timeout': 4000,
})

def fetch_ohlcv_ccxt(symbol: str, timeframe: str = "1d", 
                     since: Optional[int] = None, limit: int = 365*2) -> pd.DataFrame:
    """Fetch OHLCV from Binance using ccxt. Fetches up to 6 years of history."""
    all_rows = []
    since_param = since
    
    # Calculate 10 years ago in milliseconds if no since parameter provided
    if since_param is None:
        since_ = datetime.now().timestamp() - (10 * 365 * 24 * 60 * 60)
        since_param = int(since_ * 1000)
    
    while True:
        try:
            chunk = exchange.fetch_ohlcv(symbol, timeframe=timeframe, 
                                        since=since_param, limit=limit)
        except ccxt.BaseError as e:
            print(f"Error: {e}, retrying...")
            time.sleep(2)
            continue
            
        if not chunk or len(chunk) < limit:
            all_rows.extend(chunk if chunk else [])
            break
            
        all_rows.extend(chunk)
        since_param = chunk[-1][0] + 1
    
    if not all_rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.set_index('datetime').drop(columns=['ts'])
    return df.sort_index()

def save_to_cache(symbol: str, df: pd.DataFrame, timeframe: str = "1d"):
    """Save OHLCV to parquet cache."""
    path = DATA_RAW / f"{symbol.replace('/', '-')}__{timeframe}.parquet"
    df.to_parquet(path)

def load_from_cache(symbol: str, timeframe: str = "1d") -> Optional[pd.DataFrame]:
    """Load from parquet cache if exists."""
    path = DATA_RAW / f"{symbol.replace('/', '-')}__{timeframe}.parquet"
    return pd.read_parquet(path) if path.exists() else None

def fetch_ohlcv_cached(symbol: str, timeframe: str = "1d", 
                       force_refresh: bool = False) -> pd.DataFrame:
    """Load from cache or fetch fresh data."""
    if not force_refresh:
        cached = load_from_cache(symbol, timeframe)
        if cached is not None:
            print(f"Loaded {symbol} from cache")
            return cached
    
    print(f"Fetching {symbol} from exchange")
    df = fetch_ohlcv_ccxt(symbol, timeframe)
    if not df.empty:
        save_to_cache(symbol, df, timeframe)
    return df

def fetch_ohlcv_date_range(symbol: str, start_date: datetime, end_date: datetime, 
                           timeframe: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV from cache and filter by date range."""
    print(f"Fetching {symbol} for date range {start_date} to {end_date}")
    
    # Load full data from cache or fetch
    df = fetch_ohlcv_cached(symbol, timeframe)
    
    if df.empty:
        return df
    
    # Convert dates to Timestamp for filtering
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    
    # Filter by date range
    df_filtered = df.loc[start_ts:end_ts]
    
    return df_filtered

def get_top_symbols(n: int = 10, quote: str = "USDT") -> List[str]:
    """Get top-n trading symbols by volume."""
    try:
        markets = exchange.fetch_markets()
        filtered = [m for m in markets if m.get("quote") == quote and m.get("active")]
        filtered.sort(key=lambda m: float(m.get("info", {}).get("quoteVolume", 0) or 0), reverse=True)
        return [m["symbol"] for m in filtered][:n]
    except Exception:
        return ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT", "SOL/USDT", "DOGE/USDT"][:n]
