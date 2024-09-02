import click
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from load_log import load_ground_truth_from_bag
from scipy.signal import savgol_filter


def wrap_continuous(val):
    wrap = lambda x: np.mod(x + np.pi, 2 * np.pi) - np.pi
    dval = np.diff(val)
    dval = wrap(dval)
    retval = np.hstack([0, np.cumsum(dval)]) + val[0]
    return retval


def parse_log(bag):
    log = load_ground_truth_from_bag(bag)

    # Initialize state
    t = log[:, 0] - log[0, 0]
    t = np.linspace(0, t[-1], t.shape[0])
    x = log[:, 1]
    y = log[:, 2]
    h = savgol_filter(wrap_continuous(log[:, 6]), 51, 2)

    steering = log[:, 7]
    throttle = log[:, 8]

    vx = savgol_filter(np.gradient(x, t), 51, 2)
    vy = savgol_filter(np.gradient(y, t), 51, 2)
    r = np.hstack([0, np.diff(h)]) / (t[1] - t[0])

    return np.vstack((t, x, y, h, steering, throttle, vx, vy, r)).T


def create_dataset(
    bags, lookback=3, split=0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = [], []

    for bag in bags:
        log = parse_log(bag)

        for i in range(log.shape[0] - lookback - 1):
            X.append(log[i : i + lookback, 4:])
            y.append(log[i + lookback, 6:])

    X = np.array(X)
    y = np.array(y)

    split_idx = int(len(X) * split)

    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    return (
        torch.Tensor(X_train).float(),
        torch.Tensor(y_train).float(),
        torch.Tensor(X_test).float(),
        torch.Tensor(y_test).float(),
    )


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.float()
        out, _ = self.lstm(x)
        out = self.fc(out)

        return out


@click.command()
@click.argument("bags", type=click.Path(exists=True), nargs=-1)
@click.option("--device", "-d", default="cuda", help="Device to train the model on")
@click.option("--lookback", "-l", default=3, help="Number of time steps to look back")
@click.option(
    "--n_epochs", "-e", default=100, help="Number of epochs to train the model"
)
def train(n_epochs: int, device: str, bags: Tuple[str], lookback: int):
    X_train, y_train, X_test, y_test = create_dataset(bags, lookback)

    model = LSTMModel(input_size=5, hidden_size=64, num_layers=2, output_size=1).to(
        device
    )
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

    train_rmses = []
    test_rmses = []

    for epoch in range(n_epochs):
        model.train()

        for X_batch, y_batch in loader:
            optimizer.zero_grad()

            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            y_pred = model(X_batch).squeeze()
            loss = loss_fn(y_pred, y_batch)

            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            model.eval()

            with torch.no_grad():
                X_train, y_train = X_train.to(device), y_train.to(device)
                X_test, y_test = X_test.to(device), y_test.to(device)

                y_pred_train = model(X_train).squeeze()
                train_loss = loss_fn(y_pred_train, y_train)

                y_pred_test = model(X_test).squeeze()
                test_loss = loss_fn(y_pred_test, y_test)

            print(
                f"Epoch {epoch} // Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}"
            )

            train_rmses.append(train_loss)
            test_rmses.append(test_loss)


if __name__ == "__main__":
    train()
