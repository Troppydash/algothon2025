import os
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sn
import glob
import copy

import sklearn.dummy
import ta.momentum
import ta.volatility
import torch
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    f1_score,
    mean_squared_error,
)
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    GridSearchCV, cross_val_score,
)
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2, InterpolationMode
import ta

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random_state = 42


class Loader:
    def load(self):
        pass


def to_binary(value, thres):
    if abs(value) < thres:
        return 0

    return 1 if value > 0 else -1


class TrainLoader(Loader):
    def load(self, instr):
        df = pd.read_csv("./prices.txt", sep="\\s+", header=None, index_col=None)
        df.index = np.arange(df.shape[0])
        df.rename(columns=lambda c: int(c), inplace=True)

        # generate dataset
        xs = []
        ys = []

        # use previous 100 days
        for day in range(51, 700 - 1):
            # prev = pcf[instr][day - 50:day].values
            prev_values = df[instr][day - 50:day].values

            closes = pd.Series(prev_values)

            # compute features
            features = []

            data = pd.DataFrame({"close": closes, "volume": [10000] * len(closes)})
            ta.add_trend_ta(data, "close", "close", "close")
            ta.add_volatility_ta(data, "close", "close", "close")
            ta.add_momentum_ta(data, "close", "close", "close", "volume")
            ta.add_others_ta(data, "close")
            data = data.fillna(0)
            data = data.drop(data.columns[47], axis=1)
            lrow = data.iloc[-1].values.flatten().tolist()

            features += lrow
            features += list(closes.pct_change().values[-30:])
            #
            # macd = ta.trend.MACD(closes)
            # diff = macd.macd().values[-10:] - macd.macd_signal().values[-10:]
            # features += list(diff)
            #
            # rsi = ta.momentum.rsi(closes).values[-10:]
            # features += [*list(rsi)]
            #
            # bb = ta.volatility.bollinger_hband_indicator(closes).values[
            #     -10:
            # ]
            # features += list(bb)
            # bb = ta.volatility.bollinger_lband_indicator(closes).values[
            #     -10:
            # ]
            # features += list(bb)
            #
            # r = ta.momentum.williams_r(closes, closes, closes).values[-10:]
            # features += list(r)
            #
            # osi = ta.momentum.stoch(closes, closes, closes).values[-10:]
            # features += list(osi)
            #
            # ma20 = df[instr][day - 50: day].rolling(20).mean().values[-1]
            # ma30 = df[instr][day - 50: day].rolling(30).mean().values[-1]
            # ma40 = df[instr][day - 50: day].rolling(40).mean().values[-1]
            # ma10 = df[instr][day - 50: day].rolling(10).mean().values[-1]
            # # features += [ma10, ma20, ma30, ma40]
            # if ma20 > ma30:
            #     features += [1]
            # else:
            #     features += [0]
            #
            # if ma10 > ma30:
            #     features += [1]
            # else:
            #     features += [0]
            #
            # if ma10 > ma20:
            #     features += [1]
            # else:
            #     features += [0]
            #
            # if ma10 > ma40:
            #     features += [1]
            # else:
            #     features += [0]
            #
            # if ma20 > ma40:
            #     features += [1]
            # else:
            #     features += [0]
            #
            # if ma30 > ma40:
            #     features += [1]
            # else:
            #     features += [0]

            # # other stocks

            y = int(1 if df[instr][day+5] > df[instr][day] else 0)
            xs.append(features)
            ys.append(y)

        return xs, ys


class Preprocessor:
    def preprocess(self, x, **kwargs):
        return x

    def augment(self, x, y, **kwargs):
        return x, y


class Model:
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


class PriceDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class StockNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(2 + 17, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.layers(x)


device = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True


class WrapperNNModel(Model):
    def __init__(self, prefix="cnn"):
        self.model = StockNN().to(device)
        self.prefix = prefix

    def fit(self, X, y, X_test, y_test):
        train_dataset = PriceDataset(X, y)
        test_dataset = PriceDataset(X_test, y_test)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=100)
        test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=100)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

        xs = []
        train_losses = []
        test_losses = []

        best_model = copy.deepcopy(self.model)
        best_loss = 100000
        best_test_loss = 0
        for epoch in range(1000):
            print(f"running epoch {epoch}")

            self.model.train()
            train_loss = 0
            train_values = 0
            train_epochs = 0
            for i, (X, y) in enumerate(train_dataloader):
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()

                pred = self.model(X)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

                train_loss += np.sqrt(loss.sum().item())
                train_values += len(y)
                train_epochs += 1

            baseline_loss = 0

            self.model.eval()
            test_loss = 0
            test_epochs = 0
            test_values = 0
            correct = 0
            for i, (X, y) in enumerate(test_dataloader):
                X, y = X.to(device), y.to(device)
                with torch.no_grad():
                    pred = self.model(X)
                    loss = criterion(pred, y)
                    # baseline_loss += np.sqrt(criterion(torch.zeros_like(y), y).item())
                    baseline_loss += 0

                    correct += (torch.max(pred, 1)[1] == y).sum().item()

                    test_loss += np.sqrt(loss.item())
                    test_values += len(y)
                    test_epochs += 1

            l = train_loss / train_epochs
            if l < best_loss:
                best_loss = l
                best_model = copy.deepcopy(self.model)
                best_test_loss = test_loss / test_epochs

            # train_loss *= 100
            # test_loss *= 100
            # baseline_loss *= 100
            print(
                f"train loss {train_loss / train_epochs:.5}, test loss {test_loss / test_epochs:.5}, test acc {correct / test_values:.5}, baseline loss {baseline_loss / test_epochs:.5}"
            )

            xs.append(epoch)
            train_losses.append(train_loss / train_epochs)
            test_losses.append(test_loss / test_epochs)

            fig, ax = plt.subplots(1, 1, figsize=(5, 3))
            ax.plot(xs, train_losses, label="train loss")
            ax.plot(xs, test_losses, label="test loss")
            ax.set_xlabel("epoch")
            ax.set_ylabel("cross entropy loss")
            fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.05), ncols=3)
            fig.tight_layout()
            fig.savefig(self.prefix + "__loss.png", bbox_inches="tight")
            plt.close(fig)

        self.model = best_model

    def save(self, file="cnn.pt"):
        torch.save(self.model.state_dict(), file)

    def predict(self, X):
        self.model.eval()
        predictions = []
        for x in X:
            x = torch.tensor(x, dtype=torch.float32).to(device)
            with torch.no_grad():
                pred = self.model(torch.unsqueeze(x, 0))
                predictions.append(pred.item())

        return np.array(predictions)


class SVMModel(Model):
    def __init__(self):
        self.clf = GridSearchCV(Pipeline([
            ('scaler', MinMaxScaler()),
            ('svc', SVC(random_state=random_state, kernel='rbf'))
        ]), {
            "svc__C": [1, 10, 100, 1000, 10000, 100000, 1000000],
        }, cv=5, scoring="accuracy", n_jobs=5, verbose=1)

    def fit(self, X, y, *args):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)


class LogModel(Model):
    def __init__(self, *args, **kwargs):
        self.clf = Pipeline([
            ('chi2', SelectKBest(k=20)),
            ('scaler', StandardScaler()),
            ('svc', LogisticRegression(max_iter=100000))
        ])

    def fit(self, X, y, *args):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def score(self, X, y):
        return self.clf.score(X, y)

    def get_params(self, deep):
        return self.clf.get_params(deep)


class RFModel(Model):
    def __init__(self):
        self.clf = Pipeline([
            ('svc', RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=8))
        ])

    def fit(self, X, y, *args):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)


class BaselineModel(Model):
    def __init__(self, *args, **kwargs):
        self.clf = Pipeline([
            ('svc', DummyClassifier())
        ])

    def fit(self, X, y, *args):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def score(self, X, y):
        return self.clf.score(X, y)

    def get_params(self, deep):
        return self.clf.get_params(deep)


def main(instr):
    train_loader = TrainLoader()

    x, y = train_loader.load(instr)
    x = x[:500]
    y = y[:500]
    t_x, v_x, t_y, v_y = train_test_split(
        x, y, test_size=2 / 10, random_state=random_state, shuffle=False
    )

    base = BaselineModel()
    model = RFModel()
    model.fit(t_x, t_y)
    base.fit(t_x, t_y)

    pred = model.predict(v_x)
    model_acc = accuracy_score(v_y, pred)


    pred = base.predict(v_x)
    base_acc = accuracy_score(v_y, pred)
    print(f"Accuracy {model_acc:0.4}, Baseline {base_acc:0.4}, Diff {model_acc - base_acc:0.4}")


    """
        model_acc = float(np.mean(cross_val_score(model.clf, x, y, cv=2)))
    base_acc = float(np.mean(cross_val_score(base.clf, x, y, cv=2)))

    """

    return model_acc - base_acc


diff = []
for i in range(50):
    print('instr', i)
    diff += [main(i)]

print(f'average diff {sum(diff) / 50:.2f}, max diff {max(diff):.2f}')
