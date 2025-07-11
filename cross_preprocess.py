import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.trend import MACD
from preprocess import extract_features, extract_features2, extract_features3

COMMISSION_RATE = 0.0005
AHEAD = 20
WINDOW_FEATURES = 5

def loadDataSet():
    df = pd.read_csv("./prices.txt", sep="\\s+", header=None, index_col=None)
    df.index = np.arange(df.shape[0])
    df.rename(columns=lambda c: int(c), inplace=True)
    return df

def extract_all(price_df: pd.DataFrame, extract_features = extract_features):
    price_df.index = list(range(price_df.shape[0]))
    concat = []

    for i in range(50):
        extracted_features = extract_features(price_df[i])
        for j in range(len(extracted_features)):
            if len(concat) - 1 < j:
                concat.append([])
            concat[j].append(extracted_features[j])
    
    return tuple(concat)

cached = {}
def get_X_current(price_df: pd.DataFrame, extracted_features: tuple, today: int):
    if today in cached:
        return cached[today]
    current = {}
    for j in range(1, WINDOW_FEATURES + 1):
        prev = today - j
        # current[f"price_{j}"] = stock_df.iloc[prev]
        for i in range(50):
            for k in range(len(extracted_features)):
                current[f"{i}_{j}_{k}"] = extracted_features[k][i][prev]
    cached[today] = current
    return current

def getX(price_df: pd.DataFrame, extracted_features: tuple, stock: int):
    price_df = price_df[stock][-50:]
    price_df.index = list(range(price_df.shape[0]))
    days = price_df.shape[0]

    # Wait until the first full window of all indicator 
    X = []
    # y is buy or sell
    current = get_X_current(price_df, extracted_features, days)
    X.append(current)

    X_df = pd.DataFrame(X)
    return X_df

def preprocessTA(price_df: pd.DataFrame, extracted_features: tuple, stock: int, start=30):
    stock_df = price_df[stock]
    stock_df.index = list(range(stock_df.shape[0]))
    days = stock_df.shape[0]

    # X = features of the last few days
    # Wait until the first full window of all indicator 
    X = []
    # y is buy or sell
    y = []

    # print(stock_df)
    # print(rsi_series)
    valid_days = range(start, days - AHEAD)
    for i in valid_days:
        current = get_X_current(price_df, extracted_features, i)
        X.append(current)
        return_pct = (stock_df[i+AHEAD] - stock_df[i]) / stock_df[i]
        if abs(return_pct) < COMMISSION_RATE:
            y.append(0)
        else:
            if return_pct > 0:
                y.append(1)
            else:
                y.append(-1)
    X_df = pd.DataFrame(X)
    y_df = pd.Series(y)
    X_df.index = list(valid_days)
    y_df.index = X_df.index
    return X_df, y_df



if __name__ == "__main__":
    all_df = loadDataSet()
    X1_df, y1_df = preprocessTA(all_df, 1)
    print(X1_df)
    print(y1_df)