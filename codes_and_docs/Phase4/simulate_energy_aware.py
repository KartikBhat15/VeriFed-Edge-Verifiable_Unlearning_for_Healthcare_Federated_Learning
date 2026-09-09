"""
Phase 4: Adaptive, energy-aware client selection.

Runs the SAME FedAvg training twice, from the same starting point, over the
same number of rounds -- once using random client selection (baseline) and
once using resource-aware selection (favors clients with more energy/compute
headroom) -- and reports total energy consumed and final model utility
(AUC on PTB-XL's official validation fold) for each. This directly produces
the "measured energy/communication savings from adaptive client selection
relative to random client selection" result promised in the project
synopsis, rather than just implementing the mechanism without measuring it.

No Ray involved (same plain-loop approach as simulate_no_ray.py).
"""

import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from data_utils import get_ptbxl_arrays, dirichlet_partition
from client import ECGClient
from aggregation import weighted_average_params
from model import ECGNet, get_model_params, set_model_params
from resource_profile import (
    generate_client_profiles, reset_energy, deplete, maybe_recharge,
    select_clients_resource_aware, select_clients_random,
)


def evaluate_on_dataset(params, X, y):
    model = ECGNet()
    set_model_params(model, params)
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        out = model(X_t)
        loss = loss_fn(out, y_t).item()
        probs = torch.sigmoid(out).numpy()
    try:
        auc = roc_auc_score(y, probs, average="macro")
    except ValueError:
        auc = float("nan")
    return loss, auc


def run_strategy(strategy_name, clients, profiles, num_rounds, fraction, X_val, y_val,
                  recharge_interval, rng_seed=0):
    """
    Runs `num_rounds` of FedAvg using either resource-aware or random client
    selection each round. Returns a dict of results for reporting.
    """
    reset_energy(profiles)
    rng = np.random.default_rng(rng_seed)
    global_params = get_model_params(ECGNet())

    total_energy_spent = 0.0
    total_clients_trained = 0
    round_log = []

    for rnd in range(1, num_rounds + 1):
        maybe_recharge(profiles, rnd, interval=recharge_interval)

        if strategy_name == "resource_aware":
            train_ids, inference_only_ids = select_clients_resource_aware(profiles, fraction=fraction)
        else:
            train_ids, inference_only_ids = select_clients_random(profiles, fraction=fraction, rng=rng)

        # ---------- train only the selected clients this round ----------
        updates = []
        for cid in train_ids:
            new_params, n, _ = clients[cid].fit(global_params, config={})
            updates.append((new_params, n))
            deplete(profiles[cid])
            total_energy_spent += profiles[cid].energy_per_round
            total_clients_trained += 1

        if updates:  # only aggregate if at least one client actually trained this round
            global_params = weighted_average_params(updates)

        round_log.append({
            "round": rnd,
            "trained": list(train_ids),
            "inference_only": list(inference_only_ids),
        })

    final_loss, final_auc = evaluate_on_dataset(global_params, X_val, y_val)

    return {
        "strategy": strategy_name,
        "final_params": global_params,
        "final_auc": final_auc,
        "final_loss": final_loss,
        "total_energy_spent": total_energy_spent,
        "avg_clients_trained_per_round": total_clients_trained / num_rounds,
        "round_log": round_log,
    }


def main(ptbxl_root, num_clients, num_rounds, fraction, recharge_interval, alpha):
    print("=" * 70)
    print("PHASE 4: Adaptive, Energy-Aware Client Selection")
    print("=" * 70)

    print("\nLoading PTB-XL ...")
    X_train, y_train, X_val, y_val, X_test, y_test, mlb = get_ptbxl_arrays(ptbxl_root)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    print(f"\nPartitioning across {num_clients} clients (Dirichlet alpha={alpha}) ...")
    client_idx = dirichlet_partition(y_train, num_clients, alpha=alpha)
    clients = []
    for i, idxs in enumerate(client_idx):
        n_val = max(1, int(0.15 * len(idxs))) if len(idxs) > 0 else 0
        val_idx, train_idx = idxs[:n_val], idxs[n_val:]
        clients.append(ECGClient(i, X_train[train_idx], y_train[train_idx],
                                  X_train[val_idx], y_train[val_idx]))

    print(f"\nGenerating simulated resource profiles for {num_clients} clients ...")
    profiles_template = generate_client_profiles(num_clients, seed=42)
    for p in profiles_template:
        print(f"  client {p.client_id}: compute_score={p.compute_score:.2f}  "
              f"energy_per_round={p.energy_per_round:.1f}")

    print(f"\n[Running BASELINE: random client selection, fraction={fraction}] ...")
    profiles_random = copy.deepcopy(profiles_template)
    result_random = run_strategy("random", clients, profiles_random, num_rounds, fraction,
                                  X_val, y_val, recharge_interval, rng_seed=0)

    print(f"\n[Running ADAPTIVE: resource-aware client selection, fraction={fraction}] ...")
    profiles_aware = copy.deepcopy(profiles_template)
    result_aware = run_strategy("resource_aware", clients, profiles_aware, num_rounds, fraction,
                                 X_val, y_val, recharge_interval, rng_seed=0)

    print("\n" + "=" * 70)
    print("COMPARISON: Random vs. Resource-Aware Client Selection")
    print("=" * 70)
    print(f"{'Metric':<38}{'Random':>15}{'Resource-Aware':>18}")
    print(f"{'Total energy spent':<38}{result_random['total_energy_spent']:>15.1f}"
          f"{result_aware['total_energy_spent']:>18.1f}")
    print(f"{'Avg clients trained/round':<38}{result_random['avg_clients_trained_per_round']:>15.2f}"
          f"{result_aware['avg_clients_trained_per_round']:>18.2f}")
    print(f"{'Final val AUC':<38}{result_random['final_auc']:>15.4f}"
          f"{result_aware['final_auc']:>18.4f}")
    print(f"{'Final val loss':<38}{result_random['final_loss']:>15.4f}"
          f"{result_aware['final_loss']:>18.4f}")

    energy_diff = result_random["total_energy_spent"] - result_aware["total_energy_spent"]
    energy_pct = (energy_diff / result_random["total_energy_spent"] * 100) if result_random["total_energy_spent"] else 0.0
    print(f"\nEnergy difference: resource-aware selection used {energy_diff:+.1f} units "
          f"({energy_pct:+.1f}%) relative to random selection, for "
          f"{result_aware['final_auc'] - result_random['final_auc']:+.4f} AUC difference.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptbxl_root", type=str, required=True)
    parser.add_argument("--num_clients", type=int, default=8)
    parser.add_argument("--num_rounds", type=int, default=15)
    parser.add_argument("--fraction", type=float, default=0.5,
                         help="Fraction of clients selected to train each round")
    parser.add_argument("--recharge_interval", type=int, default=3,
                         help="Every N rounds, all clients recharge some energy")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    main(args.ptbxl_root, args.num_clients, args.num_rounds, args.fraction,
         args.recharge_interval, args.alpha)
