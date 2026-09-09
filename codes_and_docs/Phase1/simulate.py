"""
Week 1-2 deliverable: flat FedAvg baseline on PTB-XL, simulated across N clients.
Run this AFTER downloading PTB-XL (see download_ptbxl.sh) and pointing PTBXL_ROOT below.

For a quick local sanity check without the real dataset, run test_pipeline.py instead —
it exercises this exact code path with synthetic ECG-shaped data.
"""

import argparse
import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg

from data_utils import get_ptbxl_arrays, dirichlet_partition
from client import ECGClient


def make_client_fn(client_data):
    def client_fn(context):
        cid = int(context.node_config["partition-id"])
        X_train, y_train, X_val, y_val = client_data[cid]
        return ECGClient(cid, X_train, y_train, X_val, y_val).to_client()
    return client_fn


def weighted_average(metrics):
    aucs = [num_examples * m["auc"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"auc": sum(aucs) / sum(examples) if sum(examples) else 0.0}


def main(ptbxl_root: str, num_clients: int, num_rounds: int, alpha: float):
    print("Loading PTB-XL ...")
    X_train, y_train, X_val, y_val, X_test, y_test, mlb = get_ptbxl_arrays(ptbxl_root)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    print(f"Partitioning across {num_clients} clients (Dirichlet alpha={alpha}) ...")
    client_idx = dirichlet_partition(y_train, num_clients, alpha=alpha)

    client_data = []
    for idxs in client_idx:
        # simple local val split: hold out 15% of each client's data
        n_val = max(1, int(0.15 * len(idxs)))
        val_idx, train_idx = idxs[:n_val], idxs[n_val:]
        client_data.append((
            X_train[train_idx], y_train[train_idx],
            X_train[val_idx], y_train[val_idx],
        ))
        print(f"  client with {len(train_idx)} train / {len(val_idx)} val samples")

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.simulation.start_simulation(
        client_fn=make_client_fn(client_data),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptbxl_root", type=str, required=True)
    parser.add_argument("--num_clients", type=int, default=8)
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    main(args.ptbxl_root, args.num_clients, args.num_rounds, args.alpha)
