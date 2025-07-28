import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tsa.x13 import x13_arima_select_order
from statsmodels.tsa.stattools import arma_order_select_ic
from scipy.stats import linregress

from warnings import filterwarnings

filterwarnings("ignore")
NAME = "std weight"
currentPos = np.zeros(50)
positions = []

def getMyPosition(prices):
    global currentPos, weights
    df = pd.DataFrame(prices.T, columns=np.arange(50))

    tickers = list(range(50))
    limit = [0] * 50
    for i in tickers:
        limit[i] = (10000 // df[i].values[-1])

    weights = []
    for i in range(50):
        weights.append(1 / df[i].pct_change().std())

    # normalize weight
    max_weight = max(weights)
    weights = [w / max_weight for w in weights]
    prices = df

    for i in range(50):
        weights[i] /= prices.iloc[-1, i]

    low_comp = 1000000
    for i in range(50):
        if abs(weights[i]) < 0.000001:
            continue
        low_comp = min(low_comp, int(limit[i]/abs(weights[i])))

    previous = (prices * weights).sum(1).iloc[-1]
    data = (prices * weights).sum(1).pct_change().dropna()
    # o = arma_order_select_ic(data,  max_ar=5, max_ma=0, trend='n')
    # a, b = o.bic_min_order
    # print(a)

    myModel = SARIMAX(data, order=(1, 0, 0), trend='n').fit(
        maxiter=500
    )
    esp_return = myModel.forecast().iloc[-1]

    cost = data.iloc[-300:].std() / 2
    # cost = 0.0008
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

    return np.copy(currentPos)