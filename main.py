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

# Train-test split
split = int(len(df) * 0.7)
train = df.iloc[:split]
test = df.iloc[split:]

X_train = train.drop("target", axis=1)
y_train = train["target"]

X_test = test.drop("target", axis=1)
y_test = test["target"]

# Train model
model = train_model(X_train, y_train)

# Predictions
probs = model.predict_proba(X_test)[:, 1]

# Strategy
vol_mean = df["volatility_20"].mean()
positions = generate_positions(probs, df["volatility_20"].iloc[split:], vol_mean)

# Returns
returns = close.pct_change().iloc[split:]
returns = returns.iloc[:len(positions)]

strategy_returns = positions * returns.values

# Costs
cost = 0.001
trades = np.abs(np.diff(positions, prepend=0))
strategy_returns = strategy_returns - trades * cost

# Metrics
sharpe, max_dd, cumulative = compute_metrics(strategy_returns)

print("Sharpe Ratio:", sharpe)
print("Max Drawdown:", max_dd)

# Plot
plt.plot(cumulative)
plt.title("Strategy Performance")
plt.show()