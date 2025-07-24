import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

NAME = "Test SetWithFriends Main"
currentPos = np.zeros(50)
def getMyPosition(prices):
    global currentPos

    df = pd.DataFrame(prices.T, columns=np.arange(50))

    pos = pd.read_csv('./teams/setwithfriends.dump', sep='\\s+', header=None, index_col=None)
    entry = df.shape[0] - 1000 - 1
    new_pos =  pos.iloc[entry, :]
    print(new_pos)
    return new_pos

    try:
        new_pos =  pos.iloc[entry, :]
        currentPos = np.zeros(50)
        for n in [5, 8, 25, 49]:
            currentPos[n] = new_pos[n]
        return np.copy(currentPos)
    except:
        print(entry, pos.shape)
        return np.copy(currentPos)