import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

NAME = "Prefinal submission 2"
currentPos = np.zeros(50)
start = 0
myModel = None

good_stocks = list(range(50))
# Backward selection list trained on 0:1000, tested on 1000:1500
# start 1000
# mean(PL): 95.4
# return: 0.00060
# StdDev(PL): 636.43
# annSharpe(PL): 2.37
# totDvolume: 79701910
# Score: 31.79
# good_stocks = [0, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 27, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 44, 45, 46, 47, 48, 49]

# Trained on 750:1500
# Tested on 250:750
# start 250
# mean(PL): 145.4
# return: 0.00127
# StdDev(PL): 530.87
# annSharpe(PL): 4.33
# totDvolume: 57207551
# Score: 92.28
# good_stocks = [0, 1, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 23, 24, 25, 27, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 44, 46, 47, 48, 49]

# Trained on 0:750
# Tested on 750:1250
# start 750
# mean(PL): 114.8
# return: 0.00079
# StdDev(PL): 608.92
# annSharpe(PL): 2.98
# totDvolume: 72881056
# Score: 53.92
good_stocks = [0, 1, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 44, 45, 46, 47]

def getMyPosition(prices):
    global currentPos, start, myModel
    df = pd.DataFrame(prices.T, columns=np.arange(50))

    limit = [0] * 50
    low_limit = 10000
    for i in range(50):
        limit[i] = 10000 // df[i].values[-1]
        low_limit = min(low_limit, limit[i])

    prices = df[[i for i in range(50)]]

    if start % 250 == 0:
        myModel = ARIMA(prices.sum(axis=1).pct_change(), order=(2, 0, 0)).fit()
    start += 1

    t1 = prices.sum(axis=1).pct_change().values[-1]
    t2 = prices.sum(axis=1).pct_change().values[-2]
    esp_return = (t1 * myModel.params.iloc[1] + t2 * myModel.params.iloc[2] + myModel.params.iloc[0])
    
    for i in good_stocks:
        low_limit = limit[i]

        
        if abs(esp_return) > 0.0005:
            if esp_return > 0:
                currentPos[i] = low_limit
            else:
                currentPos[i] = -low_limit
        else:
            if currentPos[i] > 0 and esp_return < 0:
                currentPos[i] = 0
            elif currentPos[i] < 0 and esp_return > 0:
                currentPos[i] = 0

    return np.copy(currentPos)