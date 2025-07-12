import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from time import time
from preprocess import AHEAD, \
    extract_features2, extract_features3
from rnn_preprocess import preprocessTA, getX

import tensorflow as tf
import keras
from keras import layers

currentPos = np.zeros(50)
EXTRACT_FEATURES = extract_features2
entered = [False] * 50

first = True

model = keras.models.Sequential()
model.add(layers.SimpleRNN(units=5, activation="relu", return_sequences=True))
model.add(layers.SimpleRNN(units=5, activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Dense(3))

models = [tf.keras.models.clone_model(model) for i in range(50)]
for i in range(len(models)):
    models[i].compile(
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer="adam",
        metrics=["accuracy"]
    )
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

    train_df = df.iloc[-700:]

    for stock in good_stocks:
        if first:
            # print(stock)
            X_df, y_df = preprocessTA(train_df, stock, extract_features=EXTRACT_FEATURES)
            X_train = X_df
            y_train = y_df
            models[stock].fit(X_train, y_train, batch_size=30, epochs=100)
        
        X_pred = getX(price_df=train_df, stock=stock, extract_features=EXTRACT_FEATURES)
        # print(X_pred.shape)
        y_pred = models[stock].predict(X_pred)[0]
        # print(y_pred)
        predict_prob = max(tf.nn.softmax(y_pred))
        y_label = LABELS[np.argmax(y_pred)]
        # print(np.argmax(y_pred), y_label)
        if y_label == 1:
            currentPos[stock] = min(limit[stock]//2 * predict_prob, limit[stock])
        elif y_label == -1:
            currentPos[stock] = max(-limit[stock]//2 * predict_prob, -limit[stock])
        else:
            currentPos[stock] = 0
    
    first = False
    end = time()
    print(f"Take: {end - start}s")
    return np.copy(currentPos)