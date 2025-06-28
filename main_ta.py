import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from time import time
from preprocess import preprocessTA, getX

currentPos = np.zeros(50)

entered = [False] * 50

randomForest = RandomForestClassifier(n_estimators=50, max_depth=7)
SKIP = 5
LABELS = [-1, 0, 1]

def getMyPosition(prices):
    global currentPos, entered
    start = time()
    if len(prices[0]) % SKIP != 0:
        return np.copy(currentPos)

    df = pd.DataFrame(prices.T, columns=np.arange(50))
    limit = [0] * 50
    for i in range(50):
        limit[i] = 10000 // df[i].values[-1]

    train_df = df.iloc[-200:]
    for stock in range(50):
        # print(stock)
        X_df, y_df = preprocessTA(train_df, stock)
        # print(stock)
        X_train = X_df.iloc[:-1]
        y_train = y_df.iloc[:-1]
        randomForest.fit(X_train, y_train)
        X_pred = getX(train_df, i)
        prob = randomForest.predict_proba(X_pred)[0]
        y_pred = LABELS[np.argmax(prob)]
        predict_prob = max(prob)
        if predict_prob < 0.45:
            y_pred = 0
        if y_pred == 1:
            currentPos[stock] = min(currentPos[stock] + limit[stock]//5, limit[stock])
        elif y_pred == -1:
            currentPos[stock] = max(currentPos[stock] - limit[stock]//5, -limit[stock])
    end = time()
    print(f"Take: {end - start}s")
    return np.copy(currentPos)