import numpy as np

def generate_positions(probs, volatility, vol_mean, long_th = 0.55, short_th = 0.45):
    positions = []

    for i in range(len(probs)):
        p = probs[i]
        vol = volatility.iloc[i]

        if p > long_th and vol < vol_mean:
            positions.append(1)
        elif p < short_th and vol < vol_mean:
            positions.append(-1)
        else:
            positions.append(0)

    return np.array(positions)