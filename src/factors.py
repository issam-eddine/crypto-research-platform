import pandas as pd
import numpy as np

def momentum(close: pd.Series, lookback: int = 21) -> pd.Series:
    """Past-return momentum signal."""
    return close.pct_change(periods=lookback).shift(1)

def mean_reversion_zscore(close: pd.Series, lookback: int = 21) -> pd.Series:
    """Z-score mean reversion signal."""
    ret = close.pct_change().fillna(0)
    mu = ret.rolling(lookback).mean()
    sigma = ret.rolling(lookback).std().replace(0, np.nan)
    z = (ret - mu) / sigma
    return -z.shift(1)  # Negative for mean-reversion

def ewma_crossover(close: pd.Series, fast_window: int = 12, 
                   slow_window: int = 26, std_window: int = 20) -> pd.Series:
    """
    EWMA Crossover Strategy
    
    Signal = (fast_ewma - slow_ewma) / rolling_std
    
    Positive signals indicate uptrend (fast EWMA above slow EWMA).
    Negative signals indicate downtrend (fast EWMA below slow EWMA).
    Magnitude is normalized by volatility.
    """
    fast_ewma = close.ewm(span=fast_window).mean()
    slow_ewma = close.ewm(span=slow_window).mean()
    rolling_std = close.rolling(window=std_window).std().replace(0, np.nan)
    
    signal = (fast_ewma - slow_ewma) / rolling_std
    return signal.shift(1).fillna(0)

def cross_sectional_rank(signals: pd.DataFrame) -> pd.DataFrame:
    """Rank signals cross-sectionally to [0,1]."""
    return signals.rank(axis=1, pct=True)
