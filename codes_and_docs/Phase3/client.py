import torch
import torch.nn as nn
import numpy as np
from flwr.client import NumPyClient
from sklearn.metrics import roc_auc_score

from model import ECGNet, get_model_params, set_model_params

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ECGClient(NumPyClient):
    def __init__(self, client_id, X_train, y_train, X_val, y_val, local_epochs=1, batch_size=32, lr=1e-3):
        self.client_id = client_id
        self.model = ECGNet().to(DEVICE)
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)
        self.X_val = torch.tensor(X_val, dtype=torch.float32)
        self.y_val = torch.tensor(y_val, dtype=torch.float32)
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr

    def get_parameters(self, config):
        return get_model_params(self.model)

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        n = len(self.X_train)
        if n == 0:
            return get_model_params(self.model), 0, {}

        for _ in range(self.local_epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb, yb = self.X_train[idx].to(DEVICE), self.y_train[idx].to(DEVICE)
                opt.zero_grad()
                out = self.model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()

        return get_model_params(self.model), n, {"client_id": self.client_id}

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)
        self.model.eval()
        loss_fn = nn.BCEWithLogitsLoss()

        if len(self.X_val) == 0:
            return 0.0, 0, {"auc": 0.0}

        with torch.no_grad():
            xb, yb = self.X_val.to(DEVICE), self.y_val.to(DEVICE)
            out = self.model(xb)
            loss = loss_fn(out, yb).item()
            probs = torch.sigmoid(out).cpu().numpy()

        try:
            auc = roc_auc_score(self.y_val.numpy(), probs, average="macro")
        except ValueError:
            auc = 0.0  # can happen if a client's val split has a missing class

        return loss, len(self.X_val), {"auc": auc}
