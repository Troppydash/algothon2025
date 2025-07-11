import numpy as np
import pandas as pd
import math
from multiprocessing import Pool
import json
import matplotlib.pyplot as plt

def rolling_train_predict(model, X_df, y_df, window=100):
    y_pred_total = []
    y_true_total = []
    days = X_df.shape[0]
    labels = [-1, 0, 1]
    for i in range(days - window - 1):
        X_train = X_df.iloc[i:i+window]
        y_train = y_df.iloc[i:i+window]
        X_test = X_df.iloc[(i+window+1):(i+window+2)]
        y_test = y_df.iloc[(i+window+1):(i+window+2)]

        model.fit(X_train, y_train)
        # print(X_test)
        # y_pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[0]
        y_pred = [labels[np.argmax(prob)]]
        predict_prob = max(prob)
        if predict_prob < 0.5:
            y_pred = [0]
        print(predict_prob, prob, y_pred)
        y_pred_total.extend(y_pred)
        y_true_total.extend(y_test)
    return y_true_total, y_pred_total

def reconstruct(returns: pd.Series, start: float):
    price = []
    price.append(start)

    for price_ret in returns:
        price.append(price[-1] * (1 + price_ret))
    return np.array(price)

def get_min(values: list, labels: list):
    mini = values[0]
    label = labels[0]

    for i in range(1, len(values)):
        if values[i] < mini:
            mini = values[i]
            label = labels[i]
    return mini, label

def dynamic_time_warping(seqA: pd.Series, seqB: pd.Series):
    n = len(seqA)
    m = len(seqB)

    dtwMatrix = [[1e10] * (m+1) for _ in range(n+1)]
    pathMatrix = [[0] * (m+1) for _ in range(n+1)]
    dtwMatrix[0][0] = 0

    for i in range(1, n + 1):
        for j in range(1, m+1):
            cost = abs(seqA.iloc[i-1] - seqB.iloc[j-1])
            mini, label = get_min([dtwMatrix[i-1][j], dtwMatrix[i][j-1], dtwMatrix[i-1][j-1]], [(i-1, j), (i, j-1), (i-1, j-1)])
            dtwMatrix[i][j] = cost + mini
            pathMatrix[i][j] = label
    return dtwMatrix[n][m], dtwMatrix, pathMatrix

def wrapper_multi(pair_list: list):
    result = []
    for pair in pair_list:
        result.append((dynamic_time_warping(pair[0], pair[1])[0], pair[2], pair[3]))
    return result
        
if __name__ == "__main__":
    data = pd.read_csv("./prices.txt", sep="\\s+", header=None, index_col=None)

    inputs = []
    for i in range(50):
        for j in range(i+1, 50):
            seqA = data[i].pct_change().dropna().iloc[-100:]
            seqB = data[j].pct_change().dropna().iloc[-100:]
            inputs.append((seqA, seqB, i, j))
    
    grouped_inputs = []
    NUM_GROUP = 15
    groups = math.ceil(len(inputs) / NUM_GROUP)
    for i in range(groups):
        grouped_inputs.append(inputs[i*NUM_GROUP:(i+1)*NUM_GROUP])
    
    with Pool(NUM_GROUP) as p:
        results = p.map(wrapper_multi, grouped_inputs)
    
    
    dict_result = {}
    for result in results:
        for pair in result:
            dict_result[(pair[1], pair[2])] = pair[0]
    with open("result.json", "w") as fp:
        json.dump(results, fp)

    minis = [(100, 0)] * 50
    for i in range(50):
        for j in range(50):
            if i == j:
                continue
            key = (min(i, j), max(i, j))
            if key not in dict_result:
                continue
            if dict_result[key] < minis[i][0]:
                minis[i] = (dict_result[key], j)
    print(minis)

    # for j in range(50):
    #     for stretch in range(1, 9):
    #         seqA = data[0].pct_change().dropna().iloc[-100:]
    #         seqB = data[j].pct_change().dropna().iloc[-100 * stretch::stretch]

    #         stdA = np.std(data[0].pct_change().dropna().iloc[:-100])
    #         stdB = np.std(data[j].pct_change().dropna().iloc[:-100])

    #         seqA /= stdA
    #         seqB /= stdB
    #         mini, dtw, pathMatrix = dynamic_time_warping(seqA, seqB)
    #         n = len(pathMatrix) - 1
    #         m = len(pathMatrix[0]) - 1
    #         path = [[0] * (n + 1) for _ in range(m + 1)]

    #         back = (n, m)
    #         while back != (0, 0):
    #             path[back[0]][back[1]] = abs(seqA.iloc[back[0]-1] - seqB.iloc[back[1] - 1])
    #             back = pathMatrix[back[0]][back[1]]
    #         if mini < 50:
    #             plt.matshow(path)
    #             plt.title(f"{j}: {mini}")
    #             plt.savefig(f"warp/{j}_{stretch}_warp.png")
