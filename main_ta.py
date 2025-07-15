import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from time import time
from preprocess import preprocessTA, preprocessTA2, \
    getX, AHEAD, \
    extract_features2, get_X_current2, extract_features3, get_X_current3, \
    extract_features4, extract_features5, get_X_current_generic
from preprocess import COMMISSION_RATE

currentPos = np.zeros(50)
previousPos = np.zeros(50)
EXTRACT_FEATURES = extract_features3
GET_X_CURRENT = get_X_current_generic
entered = [False] * 50

POS_LIMIT = 10000
FACTOR = 1

def scaled_limit(limit: int):
    return limit * 2/FACTOR

first = True
models = [RandomForestClassifier(n_estimators=150, max_depth=5, random_state=2605) for i in range(50)]
# Maybe try XGBoost ... This perform worse than 
# models = [GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.01,  random_state=2605) for i in range(50)]

# Bad idea: Too fragile on different time series
good_stocks = list(range(50))
# good_stocks = [5]
# good_stocks = [14]

# good_stocks = [0, 6, 10, 16, 19, 21, 27, 34, 36, 38, 42]
# good_stocks = [6, 9, 11, 22, 24, 46]
# 19, 20, 28, 31
LABELS = [-1, 0, 1]

startPrice = [0] * 50
def stopLoss(curPrice: int, stock: int, limit: int = scaled_limit(100)):
    # Remember to update startPrice for this
    if startPrice[stock] == 0:
        return
    
    PnL = currentPos[stock] * (curPrice - startPrice[stock])
    if PnL < -limit:
        currentPos[stock] = 0
        startPrice[stock] = 0

PnL = [0] * 50
PnLMax = [0] * 50
cashes = [0] * 50
blacklist = [False] * 50

def PnLTracking(prices: pd.DataFrame):
    global currentPos, previousPos, cashes, PnL
    curPrices = prices.iloc[-2]
    # print("Tracking: ", curPrices.to_numpy())
    posLimits = np.array([int(x) for x in POS_LIMIT / curPrices])
    currentPos = np.clip(currentPos, -posLimits, posLimits)
    deltaPos = currentPos - previousPos
    # print(deltaPos)
    
    for stock in good_stocks:
        dvol = curPrices[stock] * np.abs(deltaPos[stock])
        comm = dvol * COMMISSION_RATE
        assert comm >= 0
        dcash = comm + curPrices[stock] * deltaPos[stock]
        cashes[stock] -= dcash
        pvalue = currentPos[stock] * curPrices[stock]
        PnL[stock] = cashes[stock] + pvalue
        PnLMax[stock] = max(PnLMax[stock], PnL[stock])

        if  (PnLMax[stock] > scaled_limit(150) and PnL[stock] <= scaled_limit(100)) \
            or (PnLMax[stock] > scaled_limit(50) and PnL[stock] <= scaled_limit(0)) \
            or (PnLMax[stock] > scaled_limit(250) and PnL[stock] <= scaled_limit(150) ) \
            or (PnL[stock] <= scaled_limit(-150)) \
            or (PnL[stock] >= scaled_limit(500)):
            currentPos[stock] = 0
            blacklist[stock] = True
            cashes[stock] = 0



def getMyPositionHelper(prices):
    global currentPos, entered, first, previousPos
    start = time()
    curDay = len(prices[0])

    df = pd.DataFrame(prices.T, columns=np.arange(50))
    limit = [0] * 50
    for i in range(50):
        limit[i] = 10000 // df[i].values[-1]

    train_df = df.iloc[-400:]
    # Track PnL for black-list
    PnLTracking(df)
    previousPos = np.copy(currentPos)

    for stock in good_stocks:
        if blacklist[stock]:
            continue

        stopLoss(curPrice=train_df[stock].iloc[-1], stock=stock)
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
        if curDay % AHEAD != 0:
            continue
        X_pred = getX(train_df, stock, extract_features=EXTRACT_FEATURES, get_X_current=GET_X_CURRENT)
        prob = models[stock].predict_proba(X_pred)[0]
        y_pred = LABELS[np.argmax(prob)]
        predict_prob = max(prob)
        predict_prob = 1
        if y_pred == 1:
            currentPos[stock] = min(limit[stock]//FACTOR * predict_prob, limit[stock])
        elif y_pred == -1:
            currentPos[stock] = max(-limit[stock]//FACTOR * predict_prob, -limit[stock])
        else:
            currentPos[stock] = 0
        
        # Update startPrice for stopLoss
        startPrice[stock] = train_df[stock].iloc[-1]
    
    first = False
    end = time()
    print(f"Take: {end - start}s")
    return np.copy(currentPos)

trial = 80
def getMyPosition(prices):
    global trial, PnL, PnLMax, cashes
    # Blacklist by running first 50 timestamps without doing anything
    getMyPositionHelper(prices)
    if trial > 0:
        trial -= 1
        return np.zeros(50)
    return np.copy(currentPos)
    