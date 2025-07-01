import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.trend import MACD

COMMISSION_RATE = 0.0005
AHEAD = 20
WINDOW_FEATURES = 20

def loadDataSet():
    df = pd.read_csv("./prices.txt", sep="\\s+", header=None, index_col=None)
    df.index = np.arange(df.shape[0])
    df.rename(columns=lambda c: int(c), inplace=True)
    return df

def extract_features(stock_df: pd.DataFrame):
    # X = features of the last few days
    rsi = RSIIndicator(close=stock_df, window=10)
    rsi_series = rsi.rsi()
    
    macd = MACD(close=stock_df, window_slow=26, window_fast=12, window_sign=5)
    macd_signal = macd.macd_signal()

    stoch_osc = StochasticOscillator(close=stock_df, high=stock_df, low=stock_df, window=12, smooth_window=3)
    stoch_sign = stoch_osc.stoch_signal()

    roc = ROCIndicator(close=stock_df, window=5)
    roc_vals = roc.roc()

    williamR = WilliamsRIndicator(high=stock_df, 
                                  low=stock_df, 
                                  close=stock_df, lbp=10)
    williamR_vals = williamR.williams_r()

    return rsi_series, macd_signal, stoch_sign, roc_vals, williamR_vals

def extract_all(price_df: pd.DataFrame):
    price_df.index = list(range(price_df.shape[0]))
    rsi = []
    macd = []
    stoch = []
    roc = []
    williamR = []

    for i in range(50):
        rsi_series, macd_signal, stoch_sign, roc_vals, williamR_vals = extract_features(price_df[i])
        rsi.append(rsi_series)
        macd.append(macd_signal)
        stoch.append(stoch_sign)
        roc.append(roc_vals)
        williamR.append(williamR_vals)
    return rsi, macd, stoch, roc, williamR

cached = {}
def get_X_current(price_df: pd.DataFrame, extracted_features: tuple, today: int):
    if today in cached:
        return cached[today]
    rsi_series, macd_signal, stoch_sign, roc_vals, williamR_vals = extracted_features
    current = {}
    for j in range(1, WINDOW_FEATURES + 1):
        prev = today - j
        # current[f"price_{j}"] = stock_df.iloc[prev]
        for i in range(50):
            current[f"rsi_{i}_{j}"] = rsi_series[i][prev]
            current[f"macd_{i}_{j}"] = macd_signal[i][prev]
            current[f"stoch_{i}_{j}"] = stoch_sign[i][prev]
            current[f"roc_{i}_{j}"] = roc_vals[i][prev]
            current[f"williamR_{i}_{j}"] = williamR_vals[i][prev]
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