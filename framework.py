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
    GridSearchCV,
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


class TrainLoader(Loader):
    def load(self, instr):
        df = pd.read_csv("./prices.txt", sep="\\s+", header=None, index_col=None)
        df.index = np.arange(df.shape[0])
        df.rename(columns=lambda c: int(c), inplace=True)

        # generate dataset
        xs = []
        ys = []

        pcf = df.pct_change()

        # use previous 100 days
        for day in range(51, 750 - 1):
            prev = pcf[instr][day - 50 : day].values
            prev_values = df[instr][day - 50 : day].values

            # compute features
            features = []
            macd = ta.trend.MACD(pd.Series(prev_values))
            # diff = macd.macd_diff().values[-1]
            # if diff > 0:
            #     features += [1]
            # else:
            #     features += [0]

            rsi = ta.momentum.rsi(pd.Series(prev_values)).values[-1]
            if rsi > 70:
                features += [0, 0, 0, 1, 0]
            elif rsi > 60:
                features += [0, 0, 1, 0, 0]
            elif rsi < 30:
                features += [0, 1, 0, 0, 0]
            elif rsi < 40:
                features += [1, 0, 0, 0, 0]
            else:
                features += [0, 0, 0, 0, 1]


            bb = ta.volatility.bollinger_hband_indicator(pd.Series(prev_values)).values[
                -1
            ]
            features += [bb]
            bb = ta.volatility.bollinger_lband_indicator(pd.Series(prev_values)).values[
                -1
            ]
            features += [bb]

            vol = np.std(prev)
            features += [vol]

            ma20 = df[instr][day - 50 : day].rolling(20).mean().values[-1]
            ma30 = df[instr][day - 50 : day].rolling(30).mean().values[-1]
            ma40 = df[instr][day - 50 : day].rolling(40).mean().values[-1]
            ma10 = df[instr][day - 50 : day].rolling(10).mean().values[-1]
            if ma20 > ma30:
                features += [1]
            else:
                features += [0]

            if ma10 > ma30:
                features += [1]
            else:
                features += [0]

            if ma10 > ma20:
                features += [1]
            else:
                features += [0]

            if ma10 > ma40:
                features += [1]
            else:
                features += [0]

            if ma20 > ma40:
                features += [1]
            else:
                features += [0]
            
            if ma30 > ma40:
                features += [1]
            else:
                features += [0]


            # other stocks
            others = []
            for other in range(50):
                if other == instr:
                    continue

                others += [int(p > 0) for p in list(pcf[other][day-40:day].values)]

            y = int(pcf[instr][day] > 0)
            xs.append([int(p > 0) for p in list(prev)[-40:]] + features + others)
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
                f"train loss {train_loss / train_epochs:.5}, test loss {test_loss / test_epochs:.5}, test acc {correct/test_values:.5}, baseline loss {baseline_loss / test_epochs:.5}"
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
    def __init__(self):
        self.clf = Pipeline([
            ('chi2', SelectKBest(k=20)),
            ('scaler', StandardScaler()),
            ('svc', LogisticRegression())
        ])

    def fit(self, X, y, *args):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)




def main(instr):
    train_loader = TrainLoader()

    logger.info("loading train dataset")
    x, y = train_loader.load(instr)

    t_x, v_x, t_y, v_y = train_test_split(
        x, y, test_size=2 / 10, random_state=random_state, shuffle=True
    )
    logger.info(f"holdout split, train {len(t_y)}, val {len(v_y)}")

    # model = SVMModel()
    model = LogModel()
    model.fit(t_x, t_y, v_x, v_y)

    pred = model.predict(v_x)
    mse = accuracy_score(v_y, pred)
    logger.info(f"Holdout accuracy {mse:.4}")


for i in range(50):
    print('instr', i)
    main(i)
