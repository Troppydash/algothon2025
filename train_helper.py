import numpy as np

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
