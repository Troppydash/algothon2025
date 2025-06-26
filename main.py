import numpy as np
import pandas as pd

import statsmodels.tsa.ardl as ardl

currentPos = np.zeros(50)

entered = [False] * 50

def getMyPosition(prices):
    global currentPos, entered

    df = pd.DataFrame(prices.T, columns=np.arange(50))

    limit = [0] * 50
    for i in range(50):
        limit[i] = 10000 // df[i].values[-1]


    # port = df[49].values[-1]-0.7455*df[48].values[-1]+27.9665
    # amount = min(limit[49], limit[48]*0.7455)
    # if port > 0.6:
    #     currentPos[49] = -amount
    #     currentPos[48] = amount * 0.7455
    # elif port < -0.6:
    #     currentPos[49] = amount
    #     currentPos[48] = -amount * 0.7455
   
    vols = [38, 0, 45, 20, 19, 39, 48, 15, 24, 5, 1, 40, 13, 44, 4, 36, 49, 31, 21, 30, 34, 26, 17, 47, 16, 23, 35, 25, 46, 18, 12, 7, 3, 6, 42, 43, 10, 28, 41, 14, 33, 2, 9, 11, 27, 32, 22, 29, 8, 37]


    for i in range(50):
        # if i not in [8, 15, 35, 37, 43 ]:
        #     continue

        ma20 = df[i].rolling(1).mean().values[-1]
        ma40 = df[i].rolling(40).mean().values[-1]
        if ma20 < ma40:
            currentPos[i] = limit[i] // 4
        elif ma20 > ma40:
            currentPos[i] = -limit[i] // 4
        else:
            currentPos[i] = 0
    
    return np.copy(currentPos)