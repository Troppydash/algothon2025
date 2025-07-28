import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


currentPos = np.zeros(50)
start = 0
myModel = None
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
        myModel = ARIMA(prices.sum(axis=1).pct_change(), order=(2, 0, 0)).fit()
    start += 1

    t1 = prices.sum(axis=1).pct_change().values[-1]
    t2 = prices.sum(axis=1).pct_change().values[-2]
    esp_return = (t1 * myModel.params.iloc[1] + t2 * myModel.params.iloc[2] + myModel.params.iloc[0])
    
    for i in range(50):
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