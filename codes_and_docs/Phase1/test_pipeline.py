import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg

from data_utils import dirichlet_partition, SUPERCLASSES
from client import ECGClient
from model import ECGNet

np.random.seed(0)


def make_synthetic_ptbxl(n_samples=400, n_leads=12, n_timesteps=1000, n_classes=5):
    X = np.random.randn(n_samples, n_leads, n_timesteps).astype(np.float32)
    y = np.zeros((n_samples, n_classes), dtype=np.float32)
    for i in range(n_samples):
        n_labels = np.random.choice([1, 2], p=[0.8, 0.2])
        active = np.random.choice(n_classes, size=n_labels, replace=False)
        y[i, active] = 1.0
    return X, y


def make_client_fn(client_data):
    def client_fn(context):
        cid = int(context.node_config["partition-id"])
        X_train, y_train, X_val, y_val = client_data[cid]
        return ECGClient(cid, X_train, y_train, X_val, y_val, local_epochs=1).to_client()
    return client_fn


def weighted_average(metrics):
    aucs = [num_examples * m["auc"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"auc": sum(aucs) / sum(examples) if sum(examples) else 0.0}


def main():
    print("== Step 1: model forward pass sanity check ==")
    model = ECGNet()
    dummy = np.random.randn(4, 12, 1000).astype(np.float32)
    import torch
    out = model(torch.tensor(dummy))
    assert out.shape == (4, 5), f"unexpected output shape {out.shape}"
    print(f"OK - model output shape: {tuple(out.shape)}")

    print("\n== Step 2: synthetic PTB-XL-shaped dataset ==")
    X, y = make_synthetic_ptbxl(n_samples=400)
    print(f"X: {X.shape}, y: {y.shape}, classes: {SUPERCLASSES}")

    print("\n== Step 3: Dirichlet non-IID partition across 4 clients ==")
    num_clients = 4
    client_idx = dirichlet_partition(y, num_clients, alpha=0.5)
    for i, idxs in enumerate(client_idx):
        print(f"  client {i}: {len(idxs)} samples, label counts: {y[idxs].sum(axis=0)}")

    client_data = []
    for idxs in client_idx:
        n_val = max(1, int(0.2 * len(idxs)))
        val_idx, train_idx = idxs[:n_val], idxs[n_val:]
        client_data.append((X[train_idx], y[train_idx], X[val_idx], y[val_idx]))

    print("\n== Step 4: run 3-round Flower FedAvg simulation ==")
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    history = fl.simulation.start_simulation(
        client_fn=make_client_fn(client_data),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    print("\n== ALL CHECKS PASSED ==")
    print(history)


if __name__ == "__main__":
    main()
