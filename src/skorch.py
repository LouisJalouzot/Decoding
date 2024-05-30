import numpy as np
import torch.nn as nn
import torch.optim as optim

import wandb
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping, WandbLogger
from src.metrics import scores
from src.utils import device


class Module(nn.Module):
    def __init__(self, in_dim, hidden_size, out_dim):
        super(Module, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(p=0.2),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(p=0.2),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(p=0.2),
        )

    def forward(self, X):
        X = self.fc1(X)
        X = self.fc2(X)
        X = self.fc3(X)
        return X


def simple_MLP(
    X_train,
    Y_train,
    X_valid,
    Y_valid,
    X_test,
    Y_test,
    hidden_size=1024,
    max_epochs=100,
    patience=5,
    batch_size=1024,
    verbose=False,
):

    in_dim = X_train.shape[1]
    out_dim = Y_train.shape[1]
    config = {
        "model": "simple_MLP",
        "hidden_size": hidden_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "batch_size": batch_size,
    }
    name = "_".join([f"{key}={value}" for key, value in config.items()])
    wandb_run = wandb.init(name=name, config=config)
    model = NeuralNetRegressor(
        criterion=nn.MSELoss,
        optimizer=optim.Adam,
        max_epochs=max_epochs,
        batch_size=batch_size,
        device=device,
        callbacks=[
            ("early_stopping", EarlyStopping(patience=patience)),
            ("wandb", WandbLogger(wandb_run)),
        ],
        module=Module(in_dim, hidden_size, out_dim),
    )
    model.fit(X_train, Y_train)
    X_train = np.vstack([X_train, X_valid])
    Y_train = np.vstack([Y_train, Y_valid])

    output = {}
    for t, (X, Y) in [
        ("train", (X_train, Y_train)),
        ("test", (X_test, Y_test)),
    ]:
        Y_pred = model.predict(X)
        for key, value in scores(Y, Y_pred).items():
            output[f"{t}_{key}"] = value

    return output
