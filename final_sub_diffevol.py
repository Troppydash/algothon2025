import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

NAME = "final sub dynamic diff evol new"
currentPos = np.zeros(50)
positions = []

# Original optim:
# start 1000
# mean(PL): 50.8
# return: 0.00080
# StdDev(PL): 217.65
# annSharpe(PL): 3.69
# totDvolume: 31707611
# Score: 29.01

# start 750
# mean(PL): 57.7
# return: 0.00095
# StdDev(PL): 217.34
# annSharpe(PL): 4.20
# totDvolume: 30238012
# Score: 35.93

# start 500
# mean(PL): 49.4
# return: 0.00091
# StdDev(PL): 207.59
# annSharpe(PL): 3.76
# totDvolume: 27213820
# Score: 28.65

# New optim: -abs((df * weights).sum(1).replace(0, 0.00001).pct_change().autocorr()) + 1/4 * entropy(np.abs(weights) / np.sum(np.abs(weights))) / np.log(len(weights))
# start 1000
# mean(PL): 38.8
# return: 0.00090
# StdDev(PL): 157.17
# annSharpe(PL): 3.91
# totDvolume: 21625148
# Score: 23.12

# start 750
# mean(PL): 42.9
# return: 0.00099
# StdDev(PL): 164.09
# annSharpe(PL): 4.13
# totDvolume: 21714917
# Score: 26.48

# start 500
# mean(PL): 30.2
# return: 0.00077
# StdDev(PL): 147.74
# annSharpe(PL): 3.23
# totDvolume: 19586967
# Score: 15.40

# New optim 2: -abs((df * weights).sum(1).replace(0, 0.00001).pct_change().autocorr()) - 1/5 * entropy(np.abs(weights) / np.sum(np.abs(weights))) / np.log(len(weights))

# start 500
# mean(PL): 35.5
# return: 0.00080
# StdDev(PL): 162.74
# annSharpe(PL): 3.45
# totDvolume: 22187843
# Score: 19.22

# start 750
# mean(PL): 46.5
# return: 0.00096
# StdDev(PL): 179.09
# annSharpe(PL): 4.11
# totDvolume: 24301803
# Score: 28.62

# start 1000
# mean(PL): 43.1
# return: 0.00087
# StdDev(PL): 174.78
# annSharpe(PL): 3.90
# totDvolume: 24805307
# Score: 25.64

# New optim 3: -abs((df * weights).sum(1).replace(0, 0.00001).pct_change().autocorr()) - 1/200 * np.sum(np.abs(weights) / max(np.abs(weights)))
# start 1000
# mean(PL): 51.4
# return: 0.00055
# StdDev(PL): 322.48
# annSharpe(PL): 2.52
# totDvolume: 46355554
# Score: 19.13

# New optim 4: 0.9 + -abs((df * weights).sum(1).replace(0, 0.00001).pct_change().autocorr()) - entropy(np.abs(weights) / np.sum(np.abs(weights))) / np.log(len(weights))

# start 750
# mean(PL): 54.7
# return: 0.00095
# StdDev(PL): 202.00
# annSharpe(PL): 4.28
# totDvolume: 28674315
# Score: 34.48

# start 1000
# mean(PL): 46.9
# return: 0.00080
# StdDev(PL): 199.90
# annSharpe(PL): 3.71
# totDvolume: 29385936
# Score: 26.90


def compute_weights(df, init):
    from scipy.optimize import differential_evolution
    from scipy.stats import entropy

    def test(weights):
        # Do the opposite of L1, encourage dense position vec (cause it works better than L1, since weak stocks still contribute
        # more to mean than std)
        if abs(np.sum(np.abs(weights))) < 0.000001:
            return 2
        return 0.9 + -abs((df * weights).sum(1).replace(0, 0.00001).pct_change().autocorr()) - entropy(np.abs(weights) / np.sum(np.abs(weights))) / np.log(len(weights))

    def cb(x, f):
        print(test(x))

    if init is None:
        init = np.ones(50)

    return differential_evolution(test, bounds=[(0,1)] * 50, callback=cb, x0=init, maxiter=1000).x


weights = None

def getMyPosition(prices):
    global currentPos, weights

    df = pd.DataFrame(prices.T, columns=np.arange(50))

    if weights is None or df.shape[0] % 250 == 0:
        weights = compute_weights(df, weights)
        currentPos = np.zeros(50)


    tickers = list(range(50))
    limit = [0] * 50
    for i in tickers:
        limit[i] = (10000 // df[i].values[-1])

    low_comp = 1000000
    for i in range(50):
        if abs(weights[i]) < 0.000001:
            continue
        low_comp = min(low_comp, int(limit[i]/abs(weights[i])))

    prices = df
    weighted = (prices * weights)[tickers].sum(1)
    previous = weighted.iloc[-1]
    myModel = ARIMA(weighted.pct_change().iloc[-250:], order=(1, 0, 0)).fit()
    esp_return = myModel.forecast().values[-1]

    cost = 0.0005
    for i in tickers:
        if abs(esp_return) > cost and abs(previous) > 0.00001:
            # Previous should be positive anyway, because of the bounds
            if (previous > 0 and esp_return > 0) or (previous < 0 and esp_return < 0):
                currentPos[i] = low_comp * weights[i] 
            else:
                currentPos[i] = -low_comp * weights[i]
        else:
            if esp_return * currentPos[i] < 0:
                currentPos[i] = 0

    # positions.append([int(p) for p in np.copy(currentPos)])

    # if len(positions) > 490:
    #     pos = pd.DataFrame(positions, columns=list(range(50)))
    #     pos.to_csv('pos.dump', sep='\t', index=False,header=False)

    return np.copy(currentPos)