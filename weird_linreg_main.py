import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from warnings import filterwarnings

filterwarnings("ignore")

currentPos = np.zeros(50)
positions = []
model = None

cached = {}

NAME = "weird linreg"
def train_rf(df, record=False):
    xs = []
    ys = []
    for i in range(3, df.shape[0]):
        if i in cached:
            xs.append(cached[i][0])
            ys.append(cached[i][1])
        else:
            stds = df.pct_change().iloc[:i].std()
            prices = df.iloc[i - 1]
            true_prices = df.iloc[i]
            mprices = df.iloc[:i].mean()
            stds2 = df.iloc[:i].std()

            x, y = [
                (df).sum(1).pct_change().iloc[i - 1],
                (df / df.iloc[0]).sum(1).pct_change().iloc[i - 1],
                (df / prices).sum(1).pct_change().iloc[i - 1],
                (df / mprices).sum(1).pct_change().iloc[i - 1],
                (df / stds).sum(1).pct_change().iloc[i - 1],
                (df / stds / prices).sum(1).pct_change().iloc[i - 1],
                (df / stds / df.iloc[0]).sum(1).pct_change().iloc[i - 1],
                (df / stds2).sum(1).pct_change().iloc[i - 1],
                (df / stds2 / prices).sum(1).pct_change().iloc[i - 1],
                (df / stds2 / df.iloc[0]).sum(1).pct_change().iloc[i - 1],
            ], (df / true_prices).sum(1).pct_change().iloc[i]
            xs.append(x)
            ys.append(y)

            cached[i] = (x, y)

    if record:
        total = {}
        xs = np.array(xs)
        for i in range(xs.shape[1]):
            total[("x_" + str(i))] = xs[:, i]
        total["y"] = ys
        record_df = pd.DataFrame(total)
        print(record_df.shape)
        record_df.to_csv("record_data_weird_linreg.csv")

    model = LinearRegression()
    model.fit(xs, ys)
    return model


def getMyPosition(prices):
    global currentPos, model
    df = pd.DataFrame(prices.T, columns=np.arange(50))

    tickers = list(range(50))
    limit = [0] * 50
    for i in tickers:
        limit[i] = 10000 // df[i].values[-1]

    if model is None or df.shape[0] % 5 == 0:
        model = train_rf(df)

    print(df.shape[0])
    if df.shape[0] == 999:
        train_rf(df, record=True)


    # make dataset
    stds = df.pct_change().std()
    prices = df.iloc[-1]
    mprices = df.mean()
    stds2 = df.std()

    xs = [
        (df).sum(1).pct_change().iloc[-1],
        (df / df.iloc[0]).sum(1).pct_change().iloc[-1],
        (df / prices).sum(1).pct_change().iloc[-1],
        (df / mprices).sum(1).pct_change().iloc[-1],
        (df / stds).sum(1).pct_change().iloc[-1],
        (df / stds / prices).sum(1).pct_change().iloc[-1],
        (df / stds / df.iloc[0]).sum(1).pct_change().iloc[-1],
        (df / stds2).sum(1).pct_change().iloc[-1],
        (df / stds2 / prices).sum(1).pct_change().iloc[-1],
        (df / stds2 / df.iloc[0]).sum(1).pct_change().iloc[-1],
    ]
    esp_return = model.predict([xs])[0]
    data = (df / df.iloc[-1, :]).sum(1).pct_change().dropna()

    cost = max(data.std() / 2, 0.0005)
    for i in range(50):
        if abs(esp_return) > cost:
            if esp_return > 0:
                currentPos[i] = limit[i]
            else:
                currentPos[i] = -limit[i]
        else:
            if (esp_return) * currentPos[i] < 0:
                currentPos[i] = 0

    return np.copy(currentPos)
