import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.trend import MACD

COMMISSION_RATE = 0.0005
AHEAD = 40
WINDOW_FEATURES = 40

def loadDataSet():
    df = pd.read_csv("./prices.txt", sep="\\s+", header=None, index_col=None)
    df.index = np.arange(df.shape[0])
    df.rename(columns=lambda c: int(c), inplace=True)
    return df

def getLinGrad(y: pd.Series, window: int):
    length = y.shape[0]
    all_grads = [None] * (window - 1)
    for i in range(0, length - window + 1):
        curY = y[i:(i + window)]
        x = list(range(len(curY)))
        m, b = np.polyfit(x, curY, 1)
        all_grads.append(m)
    return all_grads

def extract_features3(stock_df: pd.DataFrame):
    # X = features of the last few days
    rsi = RSIIndicator(close=stock_df, window=10)
    rsi_series = rsi.rsi()

    rsi = RSIIndicator(close=stock_df, window=20)
    rsi_series2 = rsi.rsi()

    rsi = RSIIndicator(close=stock_df, window=30)
    rsi_series3 = rsi.rsi()
    
    macd = MACD(close=stock_df, window_slow=20, window_fast=10, window_sign=5)
    macd_signal = macd.macd_diff()

    macd = MACD(close=stock_df, window_slow=12, window_fast=6, window_sign=3)
    macd_signal2 = macd.macd_diff()

    macd = MACD(close=stock_df, window_slow=30, window_fast=15, window_sign=9)
    macd_signal3 = macd.macd_diff()

    grad1 = getLinGrad(stock_df, window=5)
    grad2 = getLinGrad(stock_df, window=10)
    grad3 = getLinGrad(stock_df, window=20)

    return rsi_series, rsi_series2, rsi_series3, \
        macd_signal, macd_signal2, macd_signal3, \
        grad1, grad2, grad3      
        


def extract_features2(stock_df: pd.DataFrame):
    # X = features of the last few days
    rsi = RSIIndicator(close=stock_df, window=10)
    rsi_series = rsi.rsi()

    rsi = RSIIndicator(close=stock_df, window=20)
    rsi_series2 = rsi.rsi()
    
    macd = MACD(close=stock_df, window_slow=20, window_fast=10, window_sign=5)
    macd_signal = macd.macd_diff()

    macd = MACD(close=stock_df, window_slow=12, window_fast=6, window_sign=3)
    macd_signal2 = macd.macd_diff()

    macd = MACD(close=stock_df, window_slow=30, window_fast=15, window_sign=9)
    macd_signal3 = macd.macd_diff()

    stoch_osc = StochasticOscillator(close=stock_df, high=stock_df, low=stock_df, window=10, smooth_window=3)
    stoch_sign = stoch_osc.stoch_signal()

    stoch_osc = StochasticOscillator(close=stock_df, high=stock_df, low=stock_df, window=20, smooth_window=10)
    stoch_sign2 = stoch_osc.stoch_signal()

    roc = ROCIndicator(close=stock_df, window=10)
    roc_vals = roc.roc()

    roc = ROCIndicator(close=stock_df, window=20)
    roc_vals2 = roc.roc()

    roc = ROCIndicator(close=stock_df, window=1)
    roc_vals3 = roc.roc()

    williamR = WilliamsRIndicator(high=stock_df, 
                                  low=stock_df, 
                                  close=stock_df, lbp=10)
    williamR_vals = williamR.williams_r()

    williamR = WilliamsRIndicator(high=stock_df, 
                                  low=stock_df, 
                                  close=stock_df, lbp=20)
    williamR_vals2 = williamR.williams_r()

    return rsi_series, rsi_series2, macd_signal, macd_signal2, macd_signal3, \
        stoch_sign, stoch_sign2, roc_vals, roc_vals2, roc_vals3, williamR_vals, williamR_vals2

def extract_features(stock_df: pd.DataFrame):
    # X = features of the last few days
    rsi = RSIIndicator(close=stock_df, window=10)
    rsi_series = rsi.rsi()
    
    macd = MACD(close=stock_df, window_slow=26, window_fast=12, window_sign=5)
    macd_signal = macd.macd_diff()

    stoch_osc = StochasticOscillator(close=stock_df, high=stock_df, low=stock_df, window=12, smooth_window=3)
    stoch_sign = stoch_osc.stoch_signal()

    roc = ROCIndicator(close=stock_df, window=5)
    roc_vals = roc.roc()

    williamR = WilliamsRIndicator(high=stock_df, 
                                  low=stock_df, 
                                  close=stock_df, lbp=10)
    williamR_vals = williamR.williams_r()

    return rsi_series, macd_signal, stoch_sign, roc_vals, williamR_vals

def get_X_current3(stock_df: pd.DataFrame, extracted_features: tuple, today: int):
    rsi_series, rsi_series2, rsi_series3, \
        macd_signal, macd_signal2, macd_signal3, \
        grad1, grad2, grad3  = extracted_features
    current = {}
    for j in range(1, WINDOW_FEATURES + 1):
        prev = today - j
        # current[f"price_{j}"] = stock_df.iloc[prev]
        current[f"rsi_{j}"] = rsi_series[prev]
        current[f"rsi2_{j}"] = rsi_series2[prev]
        current[f"rsi3_{j}"] = rsi_series3[prev]
        current[f"macd_{j}"] = macd_signal[prev]
        current[f"macd2_{j}"] = macd_signal2[prev]
        current[f"macd3_{j}"] = macd_signal3[prev]
        current[f"grad1_{j}"] = grad1[prev]
        current[f"grad2_{j}"] = grad2[prev]
        current[f"grad3_{j}"] = grad3[prev]
    return current

def get_X_current2(stock_df: pd.DataFrame, extracted_features: tuple, today: int):
    rsi_series, rsi_series2, macd_signal, macd_signal2, macd_signal3, \
        stoch_sign, stoch_sign2, roc_vals, roc_vals2, roc_vals3, williamR_vals, williamR_vals2 = extracted_features
    current = {}
    for j in range(1, WINDOW_FEATURES + 1):
        prev = today - j
        # current[f"price_{j}"] = stock_df.iloc[prev]
        current[f"rsi_{j}"] = rsi_series[prev]
        current[f"rsi2_{j}"] = rsi_series2[prev]
        current[f"macd_{j}"] = macd_signal[prev]
        current[f"macd2_{j}"] = macd_signal2[prev]
        current[f"macd3_{j}"] = macd_signal3[prev]
        current[f"stoch_{j}"] = stoch_sign[prev]
        current[f"stoch2_{j}"] = stoch_sign2[prev]
        current[f"roc_{j}"] = roc_vals[prev]
        current[f"roc2_{j}"] = roc_vals2[prev]
        current[f"roc3_{j}"] = roc_vals3[prev]
        current[f"williamR_{j}"] = williamR_vals[prev]
        current[f"williamR2_{j}"] = williamR_vals2[prev]
    return current

def get_X_current(stock_df: pd.DataFrame, extracted_features: tuple, today: int):
    rsi_series, macd_signal, stoch_sign, roc_vals, williamR_vals = extracted_features
    current = {}
    for j in range(1, WINDOW_FEATURES + 1):
        prev = today - j
        # current[f"price_{j}"] = stock_df.iloc[prev]
        current[f"rsi_{j}"] = rsi_series[prev]
        current[f"macd_{j}"] = macd_signal[prev]
        current[f"stoch_{j}"] = stoch_sign[prev]
        current[f"roc_{j}"] = roc_vals[prev]
        current[f"williamR_{j}"] = williamR_vals[prev]
    return current

def getX(price_df: pd.DataFrame, stock: int, extract_features=extract_features, get_X_current = get_X_current):
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

def preprocessTA(price_df: pd.DataFrame, stock: int, start=50, extract_features=extract_features, 
                 get_X_current = get_X_current):
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