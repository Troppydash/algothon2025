import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

# DOESN'T WORK
currentPos = np.zeros(50)
start = 0
myModel = [None] * 50
def getMyPosition(prices):
    global currentPos, start, myModel
    df = pd.DataFrame(prices.T, columns=np.arange(50))

    limit = [0] * 50
    low_limit = 10000
    for i in range(50):
        limit[i] = 10000 // df[i].values[-1]
        low_limit = min(low_limit, limit[i])

    nums = [i for i in range(50)]
    prices = df[nums]

    if start % 250 == 0:
        for i in [13, 16, 17, 19, 20, 21, 22, 23, 24, 27, 31, 32, 38, 39, 41, 42, 44, 46, 47]:
            y = prices[i].pct_change().iloc[3:].to_numpy()
            X = pd.DataFrame({"-1": prices.sum(axis=1).pct_change().iloc[1:-2].to_numpy(), 
                              "-2": prices.sum(axis=1).pct_change().iloc[2:-1].to_numpy()})
            X = sm.add_constant(X)
            model = sm.OLS(y, X)
            myModel[i] = model.fit()
    start += 1

    t1 = prices.sum(axis=1).pct_change().values[-1]
    t2 = prices.sum(axis=1).pct_change().values[-2]
    
    for i in [13, 16, 17, 19, 20, 21, 22, 23, 24, 27, 31, 32, 38, 39, 41, 42, 44, 46, 47]:
        esp_return = (t1 * myModel[i].params.iloc[1] + t2 * myModel[i].params.iloc[2] + myModel[i].params.iloc[0])
        if abs(esp_return) > 0.0005:
            if esp_return > 0:
                currentPos[i] = low_limit
            else:
                currentPos[i] = -low_limit
        else:
            if currentPos[i] > 0 and esp_return < 0:
                currentPos[i] /= 1.5
            elif currentPos[i] < 0 and esp_return > 0:
                currentPos[i] /= 1.5

    return np.copy(currentPos)