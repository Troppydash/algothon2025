import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.trend import MACD

COMMISSION_RATE = 0.0010

def loadDataSet():
    df = pd.read_csv("./prices.txt", sep="\\s+", header=None, index_col=None)
    df.index = np.arange(df.shape[0])
    df.rename(columns=lambda c: int(c), inplace=True)
    return df

def preprocessTA(price_df: pd.DataFrame, stock: int):
    stock_df = price_df[stock]
    days = stock_df.shape[0]
    rsi = RSIIndicator(close=stock_df, window=10)
    rsi_series = rsi.rsi()
    
    macd = MACD(close=stock_df, window_slow=10, window_fast=5, window_sign=5)
    macd_signal = macd.macd_signal()

    stoch_osc = StochasticOscillator(close=stock_df, high=stock_df, low=stock_df, window=7, smooth_window=3)
    stoch_sign = stoch_osc.stoch_signal()

    roc = ROCIndicator(close=stock_df, window=3)
    roc_vals = roc.roc()

    williamR = WilliamsRIndicator(high=stock_df, 
                                  low=stock_df, 
                                  close=stock_df, lbp=14)
    williamR_vals = williamR.williams_r()

    # X = features of the last 7 days
    WINDOW_FEATURES = 7
    # Wait until the first full window of all indicator 
    START = 30
    X = []
    # y is buy or sell
    AHEAD = 5
    y = []

    valid_days = range(START, days - AHEAD)
    for i in valid_days:
        current = {}
        for j in range(1, WINDOW_FEATURES + 1):
            prev = i - j
            current[f"price_{j}"] = stock_df.iloc[prev]
            current[f"rsi_{j}"] = rsi_series[prev]
            current[f"macd_{j}"] = macd_signal[prev]
            current[f"stoch_{j}"] = stoch_sign[prev]
            current[f"roc_{j}"] = roc_vals[prev]
            current[f"williamR_{j}"] = williamR_vals[prev]
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