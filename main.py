import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_data
from src.features import create_features
from src.model import train_model
from src.strategy import generate_positions
from src.metrics import compute_metrics

# Load data
close = load_data()

# Features
df = create_features(close)

# Walk-forward parameters
start_train_size = int(len(df) * 0.5)

all_positions = []
all_returns = []

cost = 0.001

for t in range(start_train_size, len(df) - 1):
    train = df.iloc[:t]
    test = df.iloc[t:t+1]

    X_train = train.drop("target", axis=1)
    y_train = train["target"]

    X_test = test.drop("target", axis=1)

    # Train model
    model, scaler = train_model(X_train, y_train)

    # Scale test
    X_test_scaled = scaler.transform(X_test)

    # Predict probability
    prob = model.predict_proba(X_test_scaled)[0, 1]

    # Volatility filter
    vol = df["volatility_20"].iloc[t]
    vol_mean = df["volatility_20"].iloc[:t].mean()

    # Position
    if prob > 0.55 and vol < vol_mean:
        position = 1
    elif prob < 0.45 and vol < vol_mean:
        position = -1
    else:
        position = 0

    all_positions.append(position)

    # Return
    ret = close.pct_change().iloc[t]
    all_returns.append(ret)

# Convert to arrays
positions = np.array(all_positions)
returns = np.array(all_returns)

# Transaction costs
trades = np.abs(np.diff(positions, prepend=0))
strategy_returns = positions * returns - trades * cost

# Benchmark (buy & hold)
benchmark_returns = returns

# Metrics
sharpe, max_dd, cumulative = compute_metrics(strategy_returns)
bench_sharpe, bench_dd, bench_cum = compute_metrics(benchmark_returns)

print("Strategy Sharpe:", sharpe)
print("Strategy Max Drawdown:", max_dd)

print("Benchmark Sharpe:", bench_sharpe)
print("Benchmark Max Drawdown:", bench_dd)

# Plot
plt.plot(cumulative, label="Strategy")
plt.plot(bench_cum, label="Buy & Hold")
plt.legend()
plt.title("Strategy vs Benchmark")
plt.show()