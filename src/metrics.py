import numpy as np

def compute_metrics(strategy_returns):
    mean_return = np.mean(strategy_returns)
    std_return = np.std(strategy_returns)

    sharpe = (mean_return / std_return) * np.sqrt(252)

    cumulative = np.cumprod(1 + strategy_returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak

    max_dd = np.min(drawdown)

    return sharpe, max_dd, cumulative