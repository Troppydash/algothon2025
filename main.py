import numpy as np
import pandas as pd
import ta
import statsmodels.tsa.ardl as ardl
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

currentPos = np.zeros(50)


class RFModel:
    def __init__(self):
        self.clf = Pipeline([
            ('svc', RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=8))
        ])

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)


def make_features(df, instr):
    prev_values = df[instr][-50:].values

    closes = pd.Series(prev_values)

    # compute features
    features = []

    data = pd.DataFrame({"close": closes, "volume": [10000] * len(closes)})
    ta.add_trend_ta(data, "close", "close", "close")
    ta.add_volatility_ta(data, "close", "close", "close")
    # ta.add_momentum_ta(data, "close", "close", "close", "volume")
    # ta.add_others_ta(data, "close")
    data = data.fillna(0)
    data = data.drop(data.columns[47], axis=1)
    lrow = data.iloc[-1].values.flatten().tolist()

    features += lrow
    features += list(closes.pct_change().values[-10:])

    return features


features = {}
def make_prediction(df, instr):
    global features
    x = []
    y = []

    ahead = 5
    for i in range(50, df.shape[0] - ahead):
        if (i, instr) in features:
            xx = features[i, instr]
        else:
            xx = make_features(df.loc[:i, :], instr)
            features[i, instr] = xx

        x.append(xx)
        y.append(1 if df[instr].iloc[i + ahead] > df[instr].iloc[i] else 0)


    clf = RFModel()
    clf.fit(x, y)

    i = df.shape[0]
    if (i, instr) in features:
        f = features[i, instr]
    else:
        f = make_features(df.loc[:i, :], instr)
        features[i, instr] = f

    out = clf.predict([f])
    return out[0]

tick = 0

def getMyPosition(prices):
    global currentPos, entered, tick

    df = pd.DataFrame(prices.T, columns=np.arange(50))

    limit = [0] * 50
    for i in range(50):
        limit[i] = 10000 // df[i].values[-1]

    # port = df[49].values[-1] - 0.7455 * df[48].values[-1] + 27.9665
    # amount = min(limit[49], int(limit[48] * 0.7455))
    # if port > 0.6:
    #     currentPos[49] = -amount
    #     currentPos[48] = amount * 0.7455
    # elif port < -0.6:
    #     currentPos[49] = amount
    #     currentPos[48] = -amount * 0.7455

    if tick % 20 == 0:
        for i in range(50):
            up = make_prediction(df, i)
            if up:
                currentPos[i] = limit[i] // 2
            else:
                currentPos[i] = -limit[i] // 2

    tick += 1


    # vols = [38, 0, 45, 20, 19, 39, 48, 15, 24, 5, 1, 40, 13, 44, 4, 36, 49, 31, 21, 30, 34, 26, 17, 47, 16, 23, 35, 25, 46, 18, 12, 7, 3, 6, 42, 43, 10, 28, 41, 14, 33, 2, 9, 11, 27, 32, 22, 29, 8, 37]
    #
    #
    # for i in range(50):
    #     # if i not in [8, 15, 35, 37, 43 ]:
    #     #     continue
    #
    #     ma20 = df[i].rolling(1).mean().values[-1]
    #     ma40 = df[i].rolling(40).mean().values[-1]
    #     if ma20 < ma40:
    #         currentPos[i] = limit[i] // 4
    #     elif ma20 > ma40:
    #         currentPos[i] = -limit[i] // 4
    #     else:
    #         currentPos[i] = 0

    return np.copy(currentPos)
