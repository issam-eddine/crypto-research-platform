import pandas as pd
from typing import Dict
from pathlib import Path

DATA_PROCESSED = Path("data/processed")
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

def clean_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize price data."""
    if df.empty:
        return df
    
    df = df.copy()
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df

def align_universe(dfs: Dict[str, pd.DataFrame], freq: str = "1D") -> pd.DataFrame:
    """Align timestamps across symbols and build close prices matrix."""
    closes = {}
    
    for symbol, df in dfs.items():
        dfc = clean_price_df(df)
        if dfc.empty or "close" not in dfc.columns:
            continue
        
        dfc = dfc.resample(freq).last().ffill()
        closes[symbol] = dfc["close"]
    
    if not closes:
        return pd.DataFrame()
    
    closes_df = pd.concat(closes.values(), axis=1, keys=closes.keys())
    return closes_df.dropna(axis=1, how="all")
