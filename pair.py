import numpy as np
import pandas as pd

currentPos = np.zeros(50)

import itertools
import statsmodels.api as sm


def reg(df, stock1, stock2):
    X = df[stock1].values
    y = df[stock2].values

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    alpha, beta = model.params
    res = y - (alpha + beta * X[:, 1])

    adf_result = sm.tsa.stattools.adfuller(res)
    return adf_result[0], adf_result[1], beta, alpha


def get_pairs(df):
    stocks = pd.DataFrame(itertools.combinations(list(range(50)), 2))
    stocks.columns = ["Stock1", "Stock2"]
    stocks["correlation"] = stocks.apply(
        lambda row: np.corrcoef(df[row["Stock1"]], df[row["Stock2"]])[0, 1], axis=1
    )
    filtered_stocks = stocks[stocks.correlation > 0.85]
    filtered_stocks[["adf", "pvalue", "beta", "alpha"]] = filtered_stocks.apply(
        lambda row: pd.Series(reg(df, row["Stock1"], row["Stock2"])), axis=1
    )

    selected_stocks = filtered_stocks[filtered_stocks["pvalue"] < 0.02].sort_values(by='adf')
    return selected_stocks


selected = None
tick = 0

def getMyPosition(prices):
    global currentPos, selected, tick

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

    if tick % 10 == 0:
        currentPos = np.zeros(50)
        selected = get_pairs(df)

    tick += 1

    # handle pairs
    used = set()
    for _, row in selected.iterrows():
        x, y = int(row["Stock1"]), int(row["Stock2"])
        if x in used or y in used:
            continue

        used.add(x)
        used.add(y)

        print(f'pair {x} {y}')

        z = df[x].values[-1] * row["beta"] + row["alpha"] - df[y].values[-1]
        units = min(limit[y], limit[x] * row["beta"])
        if z > 2.5:
            currentPos[y] = int(units)
            currentPos[x] = -int(units * row["beta"])
        elif z < -2.5:
            currentPos[y] = -int(units)
            currentPos[x] = int(units * row["beta"])

    return np.copy(currentPos)
