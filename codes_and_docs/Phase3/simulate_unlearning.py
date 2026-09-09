"""
Phase 3 on REAL PTB-XL data.

Same pipeline as simulate_unlearning.py (commitment -> MIA -> unlearn ->
commitment -> MIA -> certificate -> verify -> tamper test), but instead of
training a small throwaway model on synthetic data, this:

  1. Trains a real flat FedAvg model on real PTB-XL data (same no-Ray
     approach as simulate_no_ray.py -- reuses ECGClient + aggregation.py
     directly, no Flower/Ray simulation involved).
  2. Picks one real client to be the "targe+t" (the one requesting deletion).
  3. Uses PTB-XL's own official held-out test fold (X_test/y_test) as the
     MIA non-member set -- genuinely unseen data, never touched during
     training, which is the correct rigorous choice (more correct than the
     freshly-generated synthetic non-member data the earlier version used).
  4. Runs the full unlearning + verification + certificate pipeline on the
     REAL trained model.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from data_utils import get_ptbxl_arrays, dirichlet_partition
from client import ECGClient
from aggregation import weighted_average_params
from model import ECGNet, get_model_params, set_model_params
from commitment import build_merkle_root, commitment_summary
from mia import mia_success_rate
from unlearning import gradient_ascent_unlearn
from certificate import generate_server_keypair, issue_certificate, verify_certificate


def evaluate_on_dataset(params, X, y):
    """
    Evaluate a model against an arbitrary (X, y) dataset -- unlike
    ECGClient.evaluate(), which is bound to one specific client's own small
    data split, this works against any dataset, e.g. PTB-XL's full official
    validation fold. Using a large, properly representative set like that
    avoids the "only one class present" AUC warning that a tiny per-client
    split can trigger.
    """
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


def train_flat_fedavg(clients, num_rounds):
    """Same no-Ray training loop as simulate_no_ray.py."""
    global_params = get_model_params(ECGNet())
    for rnd in range(1, num_rounds + 1):
        updates = []
        for client in clients:
            new_params, n, _ = client.fit(global_params, config={})
            updates.append((new_params, n))
        global_params = weighted_average_params(updates)

        total_loss, total_auc, total_n = 0.0, 0.0, 0
        for client in clients:
            loss, n, metrics = client.evaluate(global_params, config={})
            if n == 0:
                continue
            total_loss += loss * n
            total_auc += metrics.get("auc", 0.0) * n
            total_n += n
        avg_loss = total_loss / total_n if total_n else 0.0
        avg_auc = total_auc / total_n if total_n else 0.0
        print(f"  [ROUND {rnd}/{num_rounds}] eval_loss={avg_loss:.4f}  eval_auc={avg_auc:.4f}")

    return global_params


def main(ptbxl_root, num_clients, num_rounds, target_client, alpha):
    print("=" * 70)
    print("PHASE 3 ON REAL PTB-XL: Verifiable Unlearning")
    print("=" * 70)

    print("\n[Loading PTB-XL ...]")
    X_train, y_train, X_val, y_val, X_test, y_test, mlb = get_ptbxl_arrays(ptbxl_root)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    if not (0 <= target_client < num_clients):
        raise ValueError(f"--target_client must be between 0 and {num_clients - 1}")

    print(f"\n[1] Partitioning across {num_clients} clients (Dirichlet alpha={alpha}) ...")
    client_idx = dirichlet_partition(y_train, num_clients, alpha=alpha)

    clients = []
    client_train_data = []  # keep raw (X, y) per client for the unlearning step later
    for i, idxs in enumerate(client_idx):
        n_val = max(1, int(0.15 * len(idxs))) if len(idxs) > 0 else 0
        val_idx, train_idx = idxs[:n_val], idxs[n_val:]
        cX, cy = X_train[train_idx], y_train[train_idx]
        clients.append(ECGClient(i, cX, cy, X_train[val_idx], y_train[val_idx]))
        client_train_data.append((cX, cy))
        print(f"  client {i}: {len(train_idx)} train / {len(val_idx)} val samples"
              f"{'  <-- TARGET (will request deletion)' if i == target_client else ''}")

    print(f"\n[2] Training real flat FedAvg model for {num_rounds} rounds on real PTB-XL data ...")
    global_params = train_flat_fedavg(clients, num_rounds)
    print("    Training complete.")

    # ---------- member / non-member sets for MIA ----------
    target_X, target_y = client_train_data[target_client]
    # PTB-XL's own official held-out test fold: genuinely unseen, never used
    # in training -- the correct non-member reference set.
    nonmember_X, nonmember_y = X_test, y_test

    # ---------- pre-unlearning commitment + MIA baseline ----------
    pre_commitment = build_merkle_root(global_params)
    print(f"\n[3] Pre-unlearning commitment: {commitment_summary(global_params, 'root')}")

    model_before = ECGNet()
    set_model_params(model_before, global_params)
    mia_before = mia_success_rate(model_before, target_X, target_y, nonmember_X, nonmember_y)
    print(f"    MIA success rate on client {target_client}'s data BEFORE unlearning: {mia_before:.3f}")
    print(f"    (measured against PTB-XL's real held-out test fold as the non-member reference)")

    # ---------- deletion request -> gradient-ascent unlearning ----------
    print(f"\n[4] Deletion request received for client {target_client}. "
          f"Running gradient-ascent unlearning + retain-set stabilization...")

    retain_X = np.concatenate([client_train_data[i][0] for i in range(num_clients) if i != target_client])
    retain_y = np.concatenate([client_train_data[i][1] for i in range(num_clients) if i != target_client])

    unlearned_params = gradient_ascent_unlearn(
        ECGNet, global_params,
        forget_X=target_X, forget_y=target_y,
        retain_X=retain_X, retain_y=retain_y,
        ascent_epochs=3, ascent_lr=1e-3,
        stabilize_epochs=2, stabilize_lr=1e-3,
    )
    print("    Unlearning complete.")

    # ---------- post-unlearning commitment + MIA re-check ----------
    post_commitment = build_merkle_root(unlearned_params)
    print(f"\n[5] Post-unlearning commitment: {commitment_summary(unlearned_params, 'root')}")
    assert pre_commitment != post_commitment, \
        "BUG: commitment did not change -- unlearning had no effect on parameters!"
    print("    Confirmed: commitment changed (model parameters were actually modified).")

    model_after = ECGNet()
    set_model_params(model_after, unlearned_params)
    mia_after = mia_success_rate(model_after, target_X, target_y, nonmember_X, nonmember_y)
    print(f"    MIA success rate on client {target_client}'s data AFTER unlearning: {mia_after:.3f}")

    # ---------- accuracy impact on the rest of the model ----------
    print(f"\n[6] Checking unlearning didn't destroy overall model utility "
          f"(evaluated on PTB-XL's own official validation fold, {len(X_val)} samples) ...")
    pre_loss, pre_auc = evaluate_on_dataset(global_params, X_val, y_val)
    post_loss, post_auc = evaluate_on_dataset(unlearned_params, X_val, y_val)
    print(f"    Official val-fold AUC before unlearning: {pre_auc:.4f}  (loss={pre_loss:.4f})")
    print(f"    Official val-fold AUC after unlearning:  {post_auc:.4f}  (loss={post_loss:.4f})")

    # ---------- issue and verify a signed certificate ----------
    print(f"\n[7] Issuing signed verification certificate...")
    private_key, public_key = generate_server_keypair()
    cert = issue_certificate(
        private_key,
        client_id=target_client, round_num=num_rounds,
        pre_commitment=pre_commitment, post_commitment=post_commitment,
        mia_success_before=mia_before, mia_success_after=mia_after,
    )
    print(f"    Certificate issued: {cert.to_dict()}")

    is_valid = verify_certificate(public_key, cert)
    print(f"\n[8] Verifying certificate signature: {'VALID' if is_valid else 'INVALID'}")
    assert is_valid, "BUG: freshly issued certificate failed to verify!"

    print(f"\n[9] Tamper test: corrupting mia_success_after post-signing, re-verifying...")
    original_value = cert.mia_success_after
    cert.mia_success_after = 0.999
    is_valid_tampered = verify_certificate(public_key, cert)
    print(f"    Result: {'VALID (BUG!)' if is_valid_tampered else 'INVALID (correct)'}")
    assert not is_valid_tampered, "BUG: tampered certificate passed verification!"
    cert.mia_success_after = original_value

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED -- ON REAL PTB-XL DATA")
    print(f"  MIA success: {mia_before:.3f} -> {mia_after:.3f} (drop of {mia_before - mia_after:.3f})")
    print(f"  Commitment changed: yes")
    print(f"  Certificate signature valid: yes")
    print(f"  Tampered certificate correctly rejected: yes")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptbxl_root", type=str, required=True)
    parser.add_argument("--num_clients", type=int, default=8)
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--target_client", type=int, default=0,
                         help="Which client 'requests deletion' -- index between 0 and num_clients-1")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    main(args.ptbxl_root, args.num_clients, args.num_rounds, args.target_client, args.alpha)