from preprocess import COMMISSION_RATE, extract_features2
import tensorflow as tf
import keras
from keras import layers

import numpy as np
import pandas as pd

BEFORE = 20
AHEAD = 40
TA_WINDOW = 50

def get_X_current(extracted_feature, today: int, printable: bool = False):
    current = []
    for j in range(1, BEFORE + 1):
        prev = today - j
        all_feat = []
        for k in range(len(extracted_feature)):
            if printable:
                print(prev)
                print(extracted_feature)
            all_feat.append(extracted_feature[k][prev])
        current.append(all_feat)
    return current

def getX(price_df: pd.DataFrame, stock: int, extract_features=extract_features2, get_X_current = get_X_current):
    stock_df = price_df[stock][-(TA_WINDOW * 2 + 50):]
    stock_df.index = list(range(stock_df.shape[0]))
    days = stock_df.shape[0]

    # Wait until the first full window of all indicator 
    X = []
    # y is buy or sell
    extracted_features = extract_features(stock_df=stock_df)
    current = get_X_current(extracted_features, days)
    X.append(current)

    X_df = np.array(X)
    return X_df

# Per stock extraction
def preprocessTA(price_data: pd.DataFrame, stock: int, extract_features = extract_features2):
    length = price_data[stock].shape[0]
    price_data.index = list(range(length))
    X = []
    y = []

    extracted_feature = extract_features(price_data[stock])
    # print(rsi_series)
    # print(rsi_series.shape)
    for i in range(TA_WINDOW + BEFORE - 1, length - AHEAD):
        # Try 1 price/pct: Accuracy 0.57%, seems to do 0R
        # Try all price/pct: Overfit
        # X.append(price_data.iloc[(i - BEFORE + 1):(i+1), :].pct_change().dropna().to_numpy().reshape((BEFORE - 1, -1)))
        
        # Try feature extractions for single price
        current = get_X_current(extracted_feature=extracted_feature, today=i)
        X.append(np.array(current))
        return_pct = (price_data[stock][i+AHEAD] - price_data[stock][i]) / price_data[stock][i]
        if abs(return_pct) < COMMISSION_RATE:
            y.append(0)
        else:
            if return_pct > 0:
                y.append(1)
            else:
                y.append(-1)
    # print(X)
    X_df = np.array(X)
    y_df = np.array(y)
    y_df = tf.one_hot(y_df + 1, depth=3)
    return X_df, y_df
