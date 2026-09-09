"""
Week 3-4 deliverable: 3-tier hierarchical FedAvg on PTB-XL.

Edge Layer   -> devices train locally, never share raw data
Cluster Layer -> aggregates its own edge devices via inner FedAvg (see
                 hierarchical_client.ClusterClient)
Global Layer -> Flower's outer FedAvg strategy aggregates across clusters

Run AFTER downloading PTB-XL (see download_ptbxl.sh from the Week 1-2 deliverable).
For a quick sanity check without the real dataset, run test_hierarchical_pipeline.py.
"""

import argparse
import flwr as fl
from flwr.server.strategy import FedAvg

from data_utils import get_ptbxl_arrays, hierarchical_partition
from hierarchical_client import ClusterClient


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


def main(ptbxl_root, num_clusters, edges_per_cluster, num_global_rounds,
         cluster_alpha, edge_alpha):
    print("Loading PTB-XL ...")
    X_train, y_train, X_val, y_val, X_test, y_test, mlb = get_ptbxl_arrays(ptbxl_root)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    print(f"Building hierarchy: {num_clusters} clusters x {edges_per_cluster} edges "
          f"(cluster_alpha={cluster_alpha}, edge_alpha={edge_alpha}) ...")
    cluster_edge_idx = hierarchical_partition(
        y_train, num_clusters, edges_per_cluster,
        cluster_alpha=cluster_alpha, edge_alpha=edge_alpha,
    )

    cluster_edge_data = []
    for c, edge_idx_list in enumerate(cluster_edge_idx):
        edge_datasets = []
        for e, idxs in enumerate(edge_idx_list):
            n_val = max(1, int(0.15 * len(idxs))) if len(idxs) > 0 else 0
            val_idx, train_idx = idxs[:n_val], idxs[n_val:]
            edge_datasets.append((
                X_train[train_idx], y_train[train_idx],
                X_train[val_idx], y_train[val_idx],
            ))
            print(f"  cluster {c} / edge {e}: {len(train_idx)} train / {len(val_idx)} val samples")
        cluster_edge_data.append(edge_datasets)

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clusters,
        min_evaluate_clients=num_clusters,
        min_available_clients=num_clusters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.simulation.start_simulation(
        client_fn=make_client_fn(cluster_edge_data),
        num_clients=num_clusters,
        config=fl.server.ServerConfig(num_rounds=num_global_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptbxl_root", type=str, required=True)
    parser.add_argument("--num_clusters", type=int, default=3)
    parser.add_argument("--edges_per_cluster", type=int, default=3)
    parser.add_argument("--num_global_rounds", type=int, default=10)
    parser.add_argument("--cluster_alpha", type=float, default=0.5)
    parser.add_argument("--edge_alpha", type=float, default=5.0)
    args = parser.parse_args()
    main(args.ptbxl_root, args.num_clusters, args.edges_per_cluster,
         args.num_global_rounds, args.cluster_alpha, args.edge_alpha)
