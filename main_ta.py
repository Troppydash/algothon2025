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

    
    return np.copy(currentPos)