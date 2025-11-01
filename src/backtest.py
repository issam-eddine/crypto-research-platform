import pandas as pd
import numpy as np
from typing import Tuple, Dict

def generate_weights_from_ranks(ranks: pd.DataFrame, top_q: float = 0.9, 
                                bottom_q: float = 0.1, long_short: bool = True) -> pd.DataFrame:
    """Convert ranked signals to portfolio weights."""
    weights = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    
    for date in ranks.index:
        row = ranks.loc[date]
        longs = row[row >= row.quantile(top_q)].index.tolist()
        shorts = row[row <= row.quantile(bottom_q)].index.tolist() if long_short else []
        
        if longs:
            weights.loc[date, longs] = 1.0 / len(longs)
        if shorts:
            weights.loc[date, shorts] = -1.0 / len(shorts)
    
    return weights

def compute_portfolio_returns(weights: pd.DataFrame, price_df: pd.DataFrame,
                              transaction_cost_bps: float = 0.0) -> Tuple[pd.Series, Dict]:
    """Calculate portfolio returns with transaction costs."""
    prices = price_df.reindex(index=weights.index).ffill()
    returns = prices.pct_change().fillna(0)
    
    shifted_weights = weights.shift(1).fillna(0)
    w_prev = shifted_weights.shift(1).fillna(0)
    turnover = (shifted_weights - w_prev).abs().sum(axis=1) / 2.0
    cost_impact = turnover * transaction_cost_bps
    
    daily_returns = (shifted_weights * returns).sum(axis=1) - cost_impact
    
    return daily_returns, {"turnover_mean": turnover.mean()}

def backtest_signals(signal_df: pd.DataFrame, price_df: pd.DataFrame,
                     top_q: float = 0.9, bottom_q: float = 0.1,
                     rebalance_every: int = 21, transaction_cost_bps: float = 0.0,
                     long_short: bool = True) -> Dict:
    """High-level backtest routine."""
    ranks = signal_df.rank(axis=1, pct=True)
    weights = pd.DataFrame(0.0, index=signal_df.index, columns=signal_df.columns)
    
    for i, date in enumerate(signal_df.index):
        if i % rebalance_every == 0:
            w_row = generate_weights_from_ranks(ranks.loc[[date]], top_q, bottom_q, long_short)
            weights.loc[date] = w_row.loc[date]
    
    weights = weights.ffill().fillna(0)
    daily_returns, metrics = compute_portfolio_returns(weights, price_df, transaction_cost_bps)
    cumulative = (1 + daily_returns).cumprod()
    
    return {
        "daily_returns": daily_returns,
        "cumulative": cumulative,
        "metrics": metrics,
        "weights": weights
    }
