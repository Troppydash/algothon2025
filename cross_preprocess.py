import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.trend import MACD

COMMISSION_RATE = 0.0005
AHEAD = 5
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
    rsi = []

def get_X_current(stock_df: pd.DataFrame, extracted_features: tuple, today: int):
    rsi_series, macd_signal, stoch_sign, roc_vals, williamR_vals = extracted_features
    current = {}
    for j in range(1, WINDOW_FEATURES + 1):
        prev = today - j
        current[f"price_{j}"] = stock_df.iloc[prev]
        current[f"rsi_{j}"] = rsi_series[prev]
        current[f"macd_{j}"] = macd_signal[prev]
        current[f"stoch_{j}"] = stoch_sign[prev]
        current[f"roc_{j}"] = roc_vals[prev]
        current[f"williamR_{j}"] = williamR_vals[prev]
    return current

def getX(price_df: pd.DataFrame, stock: int):
    stock_df = price_df[stock][-50:]
    stock_df.index = list(range(stock_df.shape[0]))
    days = stock_df.shape[0]

    # Wait until the first full window of all indicator 
    X = []
    # y is buy or sell
    extracted_features = extract_features(stock_df=stock_df)
    current = get_X_current(stock_df, extracted_features, days)
    X.append(current)

    X_df = pd.DataFrame(X)
    return X_df

def preprocessTA(price_df: pd.DataFrame, stock: int, start=30):
    stock_df = price_df[stock]
    stock_df.index = list(range(stock_df.shape[0]))
    days = stock_df.shape[0]

    # X = features of the last few days
    # Wait until the first full window of all indicator 
    X = []
    extracted_features = extract_features(stock_df=stock_df)
    # y is buy or sell
    y = []

    # print(stock_df)
    # print(rsi_series)
    valid_days = range(start, days - AHEAD)
    for i in valid_days:
        current = get_X_current(stock_df, extracted_features, i)
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