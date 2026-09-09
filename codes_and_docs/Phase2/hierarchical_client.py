import torch
import torch.nn as nn
import numpy as np
from flwr.client import NumPyClient
from sklearn.metrics import roc_auc_score

from model import ECGNet, get_model_params, set_model_params
from aggregation import weighted_average_params

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClusterClient(NumPyClient):
    def __init__(self, cluster_id, edge_datasets, local_epochs=1, batch_size=32,
                 lr=1e-3, edge_rounds=1):
        self.cluster_id = cluster_id
        self.edge_datasets = edge_datasets
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.edge_rounds = edge_rounds
        self.model = ECGNet().to(DEVICE)

    def get_parameters(self, config):
        return get_model_params(self.model)

    def _train_one_edge(self, params, X_train, y_train):
        set_model_params(self.model, params)
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        n = len(X_train)
        if n == 0:
            return params, 0

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)

        for _ in range(self.local_epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb, yb = X_t[idx].to(DEVICE), y_t[idx].to(DEVICE)
                opt.zero_grad()
                out = self.model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()

        return get_model_params(self.model), n

    def fit(self, parameters, config):
        cluster_params = parameters

        for _ in range(self.edge_rounds):
            edge_updates = []
            for (X_train, y_train, _, _) in self.edge_datasets:
                if len(X_train) == 0:
                    continue
                updated_params, n = self._train_one_edge(cluster_params, X_train, y_train)
                edge_updates.append((updated_params, n))

            if not edge_updates:
                # no edge in this cluster had data this round; nothing to aggregate
                break

            cluster_params = weighted_average_params(edge_updates)

        total_examples = sum(len(X_train) for (X_train, _, _, _) in self.edge_datasets)
        return cluster_params, total_examples, {"cluster_id": self.cluster_id}

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)
        self.model.eval()
        loss_fn = nn.BCEWithLogitsLoss()

        all_val_X, all_val_y = [], []
        for (_, _, X_val, y_val) in self.edge_datasets:
            if len(X_val) > 0:
                all_val_X.append(X_val)
                all_val_y.append(y_val)

        if not all_val_X:
            return 0.0, 0, {"auc": 0.0}

        X_val = np.concatenate(all_val_X, axis=0)
        y_val = np.concatenate(all_val_y, axis=0)

        with torch.no_grad():
            xb = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
            yb = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
            out = self.model(xb)
            loss = loss_fn(out, yb).item()
            probs = torch.sigmoid(out).cpu().numpy()

        try:
            auc = roc_auc_score(y_val, probs, average="macro")
        except ValueError:
            auc = 0.0

        return loss, len(X_val), {"auc": auc, "cluster_id": self.cluster_id}
