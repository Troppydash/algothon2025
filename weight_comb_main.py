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

    # Trained on [0:1000], predicting on [1000:1500]. Terry's original optim func 
    # start 1000
    # mean(PL): 42.4
    # return: 0.00066
    # StdDev(PL): 212.13
    # annSharpe(PL): 3.16
    # totDvolume: 32148401
    # Score: 21.23
    weights = np.array([-9.99925371e-01, -2.68425677e-02, -1.17435236e-02, -9.80035456e-04,
       -2.87847866e-01, -9.98553622e-01, -1.55903512e-03, -2.25884277e-01,
       -2.90641281e-01, -1.09641324e-01, -2.60593308e-01, -1.95272802e-01,
       -4.84680823e-01, -9.99806776e-01, -1.78517533e-01, -2.25449529e-01,
       -8.01128697e-01, -9.97869278e-01, -4.59408234e-03, -9.99924677e-01,
       -9.97999150e-01, -9.98676284e-01, -2.40232548e-01, -2.75923076e-01,
       -6.06011611e-01, -2.85172169e-01, -8.89712522e-04, -9.98086931e-01,
       -7.46842409e-02, -3.11112645e-01,  7.64032022e-04, -2.17810259e-01,
       -4.48163719e-01, -9.99685881e-01, -5.66833422e-01, -6.77963156e-01,
       -4.60064461e-01, -3.67033909e-01, -9.99220016e-01, -9.99202993e-01,
       -4.60275365e-03, -4.03132792e-01, -2.59480269e-01, -3.99291175e-04,
       -5.12487731e-01, -9.98588735e-01, -3.65608970e-01, -9.99235975e-01,
       -3.93944159e-01, -9.89829421e-01]) * -1

    # weights = [0.9993368525940594, 0.14789945296769336, 0.038515145069382584, 0.0017356271571431492, 0.27343129700530877, 0.996171034494997, 0.12079461933710434, 0.23946927037250987, 0.27309640010955416, 0.1277154895419601, 0.25056077735106363, 0.2037980676858897, 0.5211630683530846, 0.9929964025320182, 0.19724839483729228, 0.3330950351355684, 0.7329854936259874, 0.9998889688236812, 0.04815774442299947, 0.9990796154425072, 0.9923260281422368, 0.9921923467957958, 0.2495801971156011, 0.29451496277749944, 0.5681381252388418, 0.3234408300580345, 0.00014230686595251285, 0.9057728895194996, 0.09097955207890607, 0.3417556750766235, -0.0015437511811802196, 0.22094171790118455, 0.4058223672504244, 0.999507807256371, 0.56983662659397, 0.6457454318548805, 0.4347793071419135, 0.3433392115959286, 0.9999938457937518, 0.9980620094348795, 0.028720048991166536, 0.4340253077760887, 0.29593490157043334, 0.0004264321313471875, 0.48023544442000476, 0.9986115004734968, 0.3484091439882988, 0.9997777824507015, 0.34927805741771545, 0.5462439143410758]
    tickers = list(range(50))
    limit = [0] * 50
    for i in tickers:
        limit[i] = (10000 // df[i].values[-1])

    # desired = [0] * 50
    # for amount in range(int(limit[0]), 0, -1):
    #     desired[0] = amount
    #     for i in tickers[1:]:
    #         # check if possible
    #         wanted = int(amount / weights[0] * weights[i])
    #         if abs(wanted) <= limit[i]:
    #             desired[i] = wanted
    #         else:
    #             break
    #     else:
    #         # ok
    #         break
    low_comp = 1000000
    for i in range(50):
        low_comp = min(low_comp, int(limit[i]/abs(weights[i])))
    

    prices = df
    weighted = (prices * weights)[tickers].sum(1)
    myModel = ARIMA(weighted.pct_change().iloc[-250:], order=(1, 0, 0)).fit()
    esp_return = myModel.forecast().values[-1]

    cost = 0.0005
    for i in tickers:
        if abs(esp_return) > cost:
            if esp_return > 0:
                currentPos[i] = low_comp * weights[i] 
            else:
                currentPos[i] = -low_comp * weights[i]
        else:
            if esp_return * currentPos[i] < 0:
                currentPos[i] = 0

    # positions.append(currentPos.tolist())

    # if len(positions) == 500:
    #     pos = pd.DataFrame(positions, columns=list(range(50)))
    #     pos.to_csv('pos.dump', sep='\t', index=False,header=False)
    return np.copy(currentPos)
