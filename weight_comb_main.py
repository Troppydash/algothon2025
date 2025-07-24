import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

NAME = "Weight_combination"
positions = []
currentPos = np.zeros(50)

def getMyPosition(prices):
    global currentPos, positions
    df = pd.DataFrame(prices.T, columns=np.arange(50))

    weights = [0.9993368525940594, 0.14789945296769336, 0.038515145069382584, 0.0017356271571431492, 0.27343129700530877, 0.996171034494997, 0.12079461933710434, 0.23946927037250987, 0.27309640010955416, 0.1277154895419601, 0.25056077735106363, 0.2037980676858897, 0.5211630683530846, 0.9929964025320182, 0.19724839483729228, 0.3330950351355684, 0.7329854936259874, 0.9998889688236812, 0.04815774442299947, 0.9990796154425072, 0.9923260281422368, 0.9921923467957958, 0.2495801971156011, 0.29451496277749944, 0.5681381252388418, 0.3234408300580345, 0.00014230686595251285, 0.9057728895194996, 0.09097955207890607, 0.3417556750766235, -0.0015437511811802196, 0.22094171790118455, 0.4058223672504244, 0.999507807256371, 0.56983662659397, 0.6457454318548805, 0.4347793071419135, 0.3433392115959286, 0.9999938457937518, 0.9980620094348795, 0.028720048991166536, 0.4340253077760887, 0.29593490157043334, 0.0004264321313471875, 0.48023544442000476, 0.9986115004734968, 0.3484091439882988, 0.9997777824507015, 0.34927805741771545, 0.5462439143410758]

    tickers = list(range(50))
    limit = [0] * 50
    for i in tickers:
        limit[i] = (10000 // df[i].values[-1])

    desired = [0] * 50
    for amount in range(int(limit[0]), 0, -1):
        desired[0] = amount
        for i in tickers[1:]:
            # check if possible
            wanted = int(amount / weights[0] * weights[i])
            if abs(wanted) <= limit[i]:
                desired[i] = wanted
            else:
                break
        else:
            # ok
            break
    

    prices = df
    weighted = (prices * weights)[tickers].sum(1)
    myModel = ARIMA(weighted.pct_change().iloc[-250:], order=(1, 0, 0)).fit()
    esp_return = myModel.forecast().values[-1]

    cost = 0.0005
    for i in tickers:
        if abs(esp_return) > cost:
            if esp_return > 0:
                currentPos[i] = int(desired[i])
            else:
                currentPos[i] = - int(desired[i])
        else:
            if esp_return * currentPos[i] < 0:
                currentPos[i] = 0

    positions.append(currentPos.tolist())

    if len(positions) == 500:
        pos = pd.DataFrame(positions, columns=list(range(50)))
        pos.to_csv('pos.dump', sep='\t', index=False,header=False)
    return np.copy(currentPos)
