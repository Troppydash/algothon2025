import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

FILE = "bidoof"
NAME = f"Test {FILE} Main"
currentPos = np.zeros(50)
def getMyPosition(prices):
    global currentPos

    df = pd.DataFrame(prices.T, columns=np.arange(50))

    pos = pd.read_csv(f'./teams/{FILE}.dump', sep='\\s+', header=None, index_col=None)
    entry = df.shape[0] - 1000 - 1

    new_pos =  pos.iloc[entry, :]
    return new_pos

    try:
        new_pos =  pos.iloc[entry, :]
        currentPos = np.zeros(50)
        for n in [5, 33, 47]:
            currentPos[n] = new_pos[n]
        return np.copy(currentPos)
    except:
        print(entry, pos.shape)
        return np.copy(currentPos)