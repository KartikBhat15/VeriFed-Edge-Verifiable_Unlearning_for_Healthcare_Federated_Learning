import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg

from data_utils import hierarchical_partition, SUPERCLASSES
from hierarchical_client import ClusterClient
from model import ECGNet

np.random.seed(0)


def make_synthetic_ptbxl(n_samples=600, n_leads=12, n_timesteps=1000, n_classes=5):
    X = np.random.randn(n_samples, n_leads, n_timesteps).astype(np.float32)
    y = np.zeros((n_samples, n_classes), dtype=np.float32)
    for i in range(n_samples):
        n_labels = np.random.choice([1, 2], p=[0.8, 0.2])
        active = np.random.choice(n_classes, size=n_labels, replace=False)
        y[i, active] = 1.0
    return X, y


def make_client_fn(cluster_edge_data):
    def client_fn(context):
        cid = int(context.node_config["partition-id"])
        edge_datasets = cluster_edge_data[cid]
        return ClusterClient(cid, edge_datasets, local_epochs=1, edge_rounds=1).to_client()
    return client_fn


def weighted_average(metrics):
    aucs = [num_examples * m["auc"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"auc": sum(aucs) / sum(examples) if sum(examples) else 0.0}


def main():
    print("== Step 1: model forward pass sanity check ==")
    model = ECGNet()
    import torch
    dummy = np.random.randn(4, 12, 1000).astype(np.float32)
    out = model(torch.tensor(dummy))
    assert out.shape == (4, 5), f"unexpected output shape {out.shape}"
    print(f"OK - model output shape: {tuple(out.shape)}")

    print("\n== Step 2: synthetic PTB-XL-shaped dataset ==")
    X, y = make_synthetic_ptbxl(n_samples=600)
    print(f"X: {X.shape}, y: {y.shape}, classes: {SUPERCLASSES}")

    print("\n== Step 3: hierarchical partition -- 3 clusters x 3 edges ==")
    num_clusters, edges_per_cluster = 3, 3
    cluster_edge_idx = hierarchical_partition(
        y, num_clusters, edges_per_cluster, cluster_alpha=0.5, edge_alpha=5.0,
    )

    cluster_edge_data = []
    for c, edge_idx_list in enumerate(cluster_edge_idx):
        edge_datasets = []
        for e, idxs in enumerate(edge_idx_list):
            n_val = max(1, int(0.2 * len(idxs))) if len(idxs) > 0 else 0
            val_idx, train_idx = idxs[:n_val], idxs[n_val:]
            edge_datasets.append((X[train_idx], y[train_idx], X[val_idx], y[val_idx]))
            print(f"  cluster {c} / edge {e}: {len(train_idx)} train / {len(val_idx)} val")
        cluster_edge_data.append(edge_datasets)

    print("\n== Step 4: run 3-round hierarchical FedAvg simulation ==")
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clusters,
        min_evaluate_clients=num_clusters,
        min_available_clients=num_clusters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    history = fl.simulation.start_simulation(
        client_fn=make_client_fn(cluster_edge_data),
        num_clients=num_clusters,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    print("\n== ALL CHECKS PASSED ==")
    print(history)


if __name__ == "__main__":
    main()
