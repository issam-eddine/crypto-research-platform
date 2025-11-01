import pandas as pd
import numpy as np
from typing import Dict

def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate annualized return."""
    mean = returns.mean()
    return (1.0 + mean) ** periods_per_year - 1.0

def annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate annualized volatility."""
    return returns.std() * (periods_per_year ** 0.5)

def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    """Calculate Sharpe ratio."""
    ar = annualized_return(returns, periods_per_year)
    av = annualized_vol(returns, periods_per_year)
    return (ar - rf) / av if av != 0 else 0.0

def max_drawdown(cum_returns: pd.Series) -> float:
    """Calculate maximum drawdown."""
    wealth = cum_returns if cum_returns.iloc[0] == 1.0 else (1 + cum_returns).cumprod()
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak
    return drawdown.min()

def compute_metrics(daily_returns: pd.Series) -> Dict[str, float]:
    """Compute all performance metrics."""
    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        return {"annual_return": 0.0, "annual_vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    
    cum = (1 + daily_returns).cumprod()
    return {
        "annual_return": annualized_return(daily_returns),
        "annual_vol": annualized_vol(daily_returns),
        "sharpe": sharpe_ratio(daily_returns),
        "max_drawdown": max_drawdown(cum)
    }
