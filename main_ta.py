import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from time import time
from preprocess import preprocessTA, getX, AHEAD, \
    extract_features2, get_X_current2, extract_features3, get_X_current3, \
    extract_features4, extract_features5, get_X_current_generic

currentPos = np.zeros(50)
EXTRACT_FEATURES = extract_features3
GET_X_CURRENT = get_X_current_generic
entered = [False] * 50

first = True
models = [RandomForestClassifier(n_estimators=150, max_depth=5, random_state=2605) for i in range(50)]
# Maybe try XGBoost ... This perform worse than 
# models = [GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.01,  random_state=2605) for i in range(50)]

# Bad idea: Too fragile on different time series
good_stocks = list(range(50))
# good_stocks = [0, 6, 10, 16, 19, 21, 27, 34, 36, 38, 42]
# good_stocks = [6, 9, 11, 22, 24, 46]
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

    train_df = df.iloc[-400:]

    for stock in good_stocks:
        if first or (curDay % 60 == 0):
            # print(stock)
            X_df, y_df = preprocessTA(train_df, stock, extract_features=EXTRACT_FEATURES, 
                                      get_X_current=GET_X_CURRENT, start=100)
            # print(stock)
            X_train = X_df
            y_train = y_df
            # print(y_train)
            # print(X_train)
            models[stock].fit(X_train, y_train)
        X_pred = getX(train_df, stock, extract_features=EXTRACT_FEATURES, get_X_current=GET_X_CURRENT)
        prob = models[stock].predict_proba(X_pred)[0]
        y_pred = LABELS[np.argmax(prob)]
        predict_prob = max(prob)
        if y_pred == 1:
            currentPos[stock] = min(limit[stock]//3 * predict_prob, limit[stock])
        elif y_pred == -1:
            currentPos[stock] = max(-limit[stock]//3 * predict_prob, -limit[stock])
    
    first = False
    end = time()
    print(f"Take: {end - start}s")
    return np.copy(currentPos)