"""
Ray-free flat FedAvg runner.

Functionally identical to simulate.py, but does NOT use
fl.simulation.start_simulation (which spins up Ray actors under the hood).
Instead, it calls ECGClient.fit()/.evaluate() directly in a plain Python
loop, aggregating with the same weighted_average_params() used in
aggregation.py for the hierarchical inner loop.

Use this if simulate.py is crashing with a Ray "Windows fatal exception:
access violation" -- that crash comes from Ray's actor system, not from our
FL logic, and removing Ray removes the crash class entirely. Trade-off:
clients train sequentially each round instead of Ray's simulated
parallelism, which does not matter for correctness, only wall-clock time.
"""

import argparse
import numpy as np
from sklearn.metrics import roc_auc_score

from data_utils import get_ptbxl_arrays, dirichlet_partition
from client import ECGClient
from aggregation import weighted_average_params
from model import ECGNet, get_model_params


def run_round(clients, global_params):
    """One FedAvg round: every client trains locally from global_params,
    updates are aggregated with weighted_average_params. Returns the new
    global params plus this round's per-client sample counts (for logging)."""
    updates = []
    for client in clients:
        new_params, n, _metrics = client.fit(global_params, config={})
        updates.append((new_params, n))
    new_global_params = weighted_average_params(updates)
    return new_global_params, [n for _, n in updates]


def run_evaluation(clients, global_params):
    """Weighted-average loss/AUC across all clients, matching the
    evaluate_metrics_aggregation_fn used in the Ray-based version."""
    total_examples = 0
    weighted_loss = 0.0
    weighted_auc = 0.0
    for client in clients:
        loss, n, metrics = client.evaluate(global_params, config={})
        if n == 0:
            continue
        total_examples += n
        weighted_loss += loss * n
        weighted_auc += metrics.get("auc", 0.0) * n
    if total_examples == 0:
        return 0.0, 0.0
    return weighted_loss / total_examples, weighted_auc / total_examples


def main(ptbxl_root, num_clients, num_rounds, alpha):
    print("Loading PTB-XL ...")
    X_train, y_train, X_val, y_val, X_test, y_test, mlb = get_ptbxl_arrays(ptbxl_root)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    print(f"Partitioning across {num_clients} clients (Dirichlet alpha={alpha}) ...")
    client_idx = dirichlet_partition(y_train, num_clients, alpha=alpha)

    clients = []
    for i, idxs in enumerate(client_idx):
        n_val = max(1, int(0.15 * len(idxs))) if len(idxs) > 0 else 0
        val_idx, train_idx = idxs[:n_val], idxs[n_val:]
        clients.append(ECGClient(
            i, X_train[train_idx], y_train[train_idx],
            X_train[val_idx], y_train[val_idx],
        ))
        print(f"  client {i}: {len(train_idx)} train / {len(val_idx)} val samples")

    global_params = get_model_params(ECGNet())

    print(f"\nStarting {num_rounds}-round Ray-free FedAvg simulation ...\n")
    for rnd in range(1, num_rounds + 1):
        global_params, sample_counts = run_round(clients, global_params)
        loss, auc = run_evaluation(clients, global_params)
        print(f"[ROUND {rnd}/{num_rounds}] samples/client={sample_counts}  "
              f"eval_loss={loss:.4f}  eval_auc={auc:.4f}")

    print("\nDone. No Ray involved at any point in this run.")
    return global_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptbxl_root", type=str, required=True)
    parser.add_argument("--num_clients", type=int, default=8)
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    main(args.ptbxl_root, args.num_clients, args.num_rounds, args.alpha)