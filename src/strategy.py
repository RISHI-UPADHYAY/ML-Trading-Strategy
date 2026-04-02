import numpy as np

def generate_positions(probs, volatility, vol_mean):
    positions = []

    for i in range(len(probs)):
        p = probs[i]
        vol = volatility.iloc[i]

        if p > 0.55 and vol < vol_mean:
            positions.append(1)
        elif p < 0.45 and vol < vol_mean:
            positions.append(-1)
        else:
            positions.append(0)

    return np.array(positions)