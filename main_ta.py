import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from time import time
from preprocess import preprocessTA, getX, AHEAD

currentPos = np.zeros(50)

entered = [False] * 50

first = True
models = [RandomForestClassifier(n_estimators=150, max_depth=3, random_state=2605) for i in range(50)]
good_stocks = [0, 6, 7, 9, 11, 22, 24, 30, 35, 46]
# 19, 20, 28, 31
LABELS = [-1, 0, 1]

def getMyPosition(prices):
    global currentPos, entered, first
    start = time()
    curDay = len(prices[0])
    if curDay % AHEAD != 0:
        return np.copy(currentPos)

    df = pd.DataFrame(prices.T, columns=np.arange(50))
    limit = [0] * 50
    for i in range(50):
        limit[i] = 10000 // df[i].values[-1]

    train_df = df.iloc[-300:]

    for stock in good_stocks:
        if first or (curDay % 15 == 0):
            # print(stock)
            X_df, y_df = preprocessTA(train_df, stock)
            # print(stock)
            X_train = X_df
            y_train = y_df
            # print(X_train)
            # print(y_train)
            models[stock].fit(X_train, y_train)
        X_pred = getX(train_df, i)
        prob = models[stock].predict_proba(X_pred)[0]
        y_pred = LABELS[np.argmax(prob)]
        predict_prob = max(prob)
        if y_pred == 1:
            currentPos[stock] = min(limit[stock]//2 * predict_prob, limit[stock])
        elif y_pred == -1:
            currentPos[stock] = max(-limit[stock]//2 * predict_prob, -limit[stock])
    
    first = False
    end = time()
    print(f"Take: {end - start}s")
    return np.copy(currentPos)