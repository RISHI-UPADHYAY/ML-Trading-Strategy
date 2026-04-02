import pandas as pd
import numpy as np

def create_features(close):
    df = pd.DataFrame()

    df["return"] = close.pct_change()
    df["log_return"] = np.log(close / close.shift(1))

    df["momentum_10"] = close.pct_change(10)
    df["momentum_20"] = close.pct_change(20)
    df["momentum_50"] = close.pct_change(50)

    df["volatility_20"] = close.rolling(20).std()

    df["z_score_20"] = (close - close.rolling(20).mean()) / close.rolling(20).std()

    df["ma_gap"] = close / close.rolling(20).mean() - 1

    # Target
    future_return = close.pct_change().shift(-1)
    df["target"] = (future_return > 0).astype(int)

    return df.dropna()