import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from time import time
from cross_preprocess import preprocessTA, getX, AHEAD, extract_all
from preprocess import extract_features, extract_features2, extract_features3, extract_features4

currentPos = np.zeros(50)

entered = [False] * 50

first = True
models = [RandomForestClassifier(n_estimators=150, max_depth=5, random_state=2605) for i in range(50)]
# Maybe try XGBoost ... This perform worse than 
# models = [GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.01,  random_state=2605) for i in range(50)]

# Bad idea: Too fragile on different time series
good_stocks = list(range(50))
# 6, 9, 11, 22, 24, 46
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
    extracted_features = extract_all(price_df=train_df, extract_features=extract_features3)

    for stock in good_stocks:
        if first or (curDay % (AHEAD * 2) == 0):
            X_df, y_df = preprocessTA(train_df, extracted_features=extracted_features, stock=stock, start=50)
            
            X_train = X_df
            y_train = y_df
            models[stock].fit(X_train, y_train)
        X_pred = getX(train_df, extracted_features=extracted_features, stock=i)
        prob = models[stock].predict_proba(X_pred)[0]
        y_pred = LABELS[np.argmax(prob)]
        predict_prob = max(prob)
        if curDay % AHEAD == 0:
            if y_pred == 1:
                currentPos[stock] = min(limit[stock]//2 * predict_prob, limit[stock])
            elif y_pred == -1:
                currentPos[stock] = max(-limit[stock]//2 * predict_prob, -limit[stock])
        # else:
        #     if (y_pred == 1 and currentPos[stock] < 0) or \
        #         (y_pred == -1 and currentPos[stock] > 0):
        #         currentPos[stock] *= (1 - 0.2 * predict_prob)
        #     if (y_pred == -1 and currentPos[stock] < 0) or \
        #         (y_pred == 1 and currentPos[stock] > 0):
        #         currentPos[stock] *= (1 + 0.2 * predict_prob)
    
    first = False
    end = time()
    print(f"Take: {end - start}s")
    return np.copy(currentPos)