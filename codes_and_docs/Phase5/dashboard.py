"""
Phase 5: VeriFed-Edge Streamlit Dashboard.

Ties together everything built in Phases 1-4 into one UI:
  - Tab 1: run real FedAvg training on PTB-XL, watch loss/AUC live
  - Tab 2: submit a deletion request for a client, watch the full
    verifiable-unlearning pipeline run (commitment -> MIA -> unlearn ->
    commitment -> MIA -> signed certificate -> verify -> tamper test)
  - Tab 3: compare random vs. resource-aware client selection

Run with:
    streamlit run dashboard.py

Reuses every library module built in prior phases directly -- this file
contains NO new FL/crypto/unlearning logic of its own, only UI code that
calls into data_utils.py, model.py, client.py, aggregation.py,
commitment.py, mia.py, unlearning.py, certificate.py, and
resource_profile.py exactly as the command-line scripts do.
"""

import copy
import pandas as pd
import streamlit as st
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
from resource_profile import (
    generate_client_profiles, reset_energy, deplete, maybe_recharge,
    select_clients_resource_aware, select_clients_random,
)

st.set_page_config(page_title="VeriFed-Edge Dashboard", layout="wide")


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


# ============================================================
# Sidebar: load dataset once, shared across all three tabs
# ============================================================
st.sidebar.header("Dataset")
ptbxl_root = st.sidebar.text_input("PTB-XL root path", value="")
num_clients = st.sidebar.slider("Number of clients", min_value=3, max_value=16, value=8)
alpha = st.sidebar.slider("Dirichlet alpha (non-IID skew)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)

if st.sidebar.button("Load PTB-XL and partition clients"):
    if not ptbxl_root:
        st.sidebar.error("Enter a PTB-XL root path first.")
    else:
        with st.spinner("Loading PTB-XL and partitioning across clients..."):
            X_train, y_train, X_val, y_val, X_test, y_test, mlb = get_ptbxl_arrays(ptbxl_root)
            client_idx = dirichlet_partition(y_train, num_clients, alpha=alpha)

            clients, client_train_data = [], []
            for i, idxs in enumerate(client_idx):
                n_val = max(1, int(0.15 * len(idxs))) if len(idxs) > 0 else 0
                val_idx, train_idx = idxs[:n_val], idxs[n_val:]
                cX, cy = X_train[train_idx], y_train[train_idx]
                clients.append(ECGClient(i, cX, cy, X_train[val_idx], y_train[val_idx]))
                client_train_data.append((cX, cy))

            st.session_state.clients = clients
            st.session_state.client_train_data = client_train_data
            st.session_state.X_val, st.session_state.y_val = X_val, y_val
            st.session_state.X_test, st.session_state.y_test = X_test, y_test
            st.session_state.num_clients = num_clients
            st.session_state.global_params = get_model_params(ECGNet())
            st.session_state.training_history = []
        st.sidebar.success(f"Loaded. {num_clients} clients ready.")

if "clients" not in st.session_state:
    st.info("Use the sidebar to load PTB-XL and partition clients before using any tab below.")
    st.stop()

st.title("VeriFed-Edge Dashboard")

tab1, tab2, tab3 = st.tabs(["1. Training", "2. Unlearning & Verification", "3. Energy-Aware Selection"])

# ============================================================
# TAB 1: Training
# ============================================================
with tab1:
    st.subheader("Flat FedAvg Training")
    num_rounds = st.number_input("Rounds to train", min_value=1, max_value=100, value=10, key="train_rounds")

    if st.button("Run training"):
        clients = st.session_state.clients
        global_params = st.session_state.global_params
        history = st.session_state.training_history
        chart_placeholder = st.empty()
        progress = st.progress(0.0)

        for rnd in range(1, num_rounds + 1):
            updates = []
            for client in clients:
                new_params, n, _ = client.fit(global_params, config={})
                updates.append((new_params, n))
            global_params = weighted_average_params(updates)

            loss, auc = evaluate_on_dataset(global_params, st.session_state.X_val, st.session_state.y_val)
            history.append({"round": len(history) + 1, "val_loss": loss, "val_auc": auc})

            df = pd.DataFrame(history).set_index("round")
            chart_placeholder.line_chart(df)
            progress.progress(rnd / num_rounds)

        st.session_state.global_params = global_params
        st.session_state.training_history = history
        st.success(f"Training complete. {len(history)} total rounds so far this session.")

    if st.session_state.training_history:
        latest = st.session_state.training_history[-1]
        col1, col2 = st.columns(2)
        col1.metric("Latest validation loss", f"{latest['val_loss']:.4f}")
        col2.metric("Latest validation AUC", f"{latest['val_auc']:.4f}")

# ============================================================
# TAB 2: Unlearning & Verification
# ============================================================
with tab2:
    st.subheader("Submit a Deletion Request")
    st.caption("Requires at least one round of training in Tab 1 first.")

    target_client = st.selectbox("Client requesting deletion", options=list(range(st.session_state.num_clients)))

    if st.button("Submit deletion request"):
        if not st.session_state.training_history:
            st.error("Train the model in Tab 1 first -- there's no trained model to unlearn from yet.")
        else:
            global_params = st.session_state.global_params
            client_train_data = st.session_state.client_train_data
            target_X, target_y = client_train_data[target_client]
            nonmember_X, nonmember_y = st.session_state.X_test, st.session_state.y_test

            with st.spinner("Committing model, measuring MIA baseline..."):
                pre_commitment = build_merkle_root(global_params)
                model_before = ECGNet()
                set_model_params(model_before, global_params)
                mia_before = mia_success_rate(model_before, target_X, target_y, nonmember_X, nonmember_y)

            with st.spinner("Running gradient-ascent unlearning..."):
                num_clients_total = st.session_state.num_clients
                retain_X_list = [client_train_data[i][0] for i in range(num_clients_total) if i != target_client]
                retain_y_list = [client_train_data[i][1] for i in range(num_clients_total) if i != target_client]
                import numpy as np
                retain_X = np.concatenate(retain_X_list)
                retain_y = np.concatenate(retain_y_list)

                unlearned_params = gradient_ascent_unlearn(
                    ECGNet, global_params,
                    forget_X=target_X, forget_y=target_y,
                    retain_X=retain_X, retain_y=retain_y,
                    ascent_epochs=3, ascent_lr=1e-3,
                    stabilize_epochs=2, stabilize_lr=1e-3,
                )

            post_commitment = build_merkle_root(unlearned_params)
            model_after = ECGNet()
            set_model_params(model_after, unlearned_params)
            mia_after = mia_success_rate(model_after, target_X, target_y, nonmember_X, nonmember_y)

            private_key, public_key = generate_server_keypair()
            cert = issue_certificate(
                private_key, client_id=target_client, round_num=len(st.session_state.training_history),
                pre_commitment=pre_commitment, post_commitment=post_commitment,
                mia_success_before=mia_before, mia_success_after=mia_after,
            )
            is_valid = verify_certificate(public_key, cert)

            st.session_state.last_cert = cert
            st.session_state.last_cert_public_key = public_key
            st.session_state.last_cert_valid = is_valid
            st.session_state.last_mia = (mia_before, mia_after)
            st.session_state.last_commitments = (pre_commitment, post_commitment)
            st.session_state.unlearned_params = unlearned_params

            st.success("Deletion request processed.")

    if "last_cert" in st.session_state:
        st.divider()
        mia_before, mia_after = st.session_state.last_mia
        col1, col2, col3 = st.columns(3)
        col1.metric("MIA success before", f"{mia_before:.3f}")
        col2.metric("MIA success after", f"{mia_after:.3f}", delta=f"{mia_after - mia_before:+.3f}")
        col3.metric("Certificate valid", "YES" if st.session_state.last_cert_valid else "NO")

        pre_c, post_c = st.session_state.last_commitments
        st.text(f"Pre-commitment:  {pre_c}")
        st.text(f"Post-commitment: {post_c}")
        st.caption("Different commitments confirm the model parameters actually changed.")

        st.json(st.session_state.last_cert.to_dict())

        if st.button("Tamper test: corrupt certificate and re-verify"):
            import copy as _copy
            tampered = _copy.deepcopy(st.session_state.last_cert)
            tampered.mia_success_after = 0.999
            still_valid = verify_certificate(st.session_state.last_cert_public_key, tampered)
            if still_valid:
                st.error("BUG: tampered certificate passed verification!")
            else:
                st.success("Correct: tampered certificate was rejected (signature verification caught it).")

# ============================================================
# TAB 3: Energy-Aware Selection
# ============================================================
with tab3:
    st.subheader("Random vs. Resource-Aware Client Selection")
    energy_rounds = st.number_input("Rounds to simulate", min_value=1, max_value=50, value=15, key="energy_rounds")
    fraction = st.slider("Fraction of clients selected per round", 0.1, 1.0, 0.5)
    recharge_interval = st.slider("Recharge every N rounds", 1, 10, 3)

    if st.button("Run energy comparison"):
        clients = st.session_state.clients
        num_clients_total = st.session_state.num_clients
        X_val, y_val = st.session_state.X_val, st.session_state.y_val

        profiles_template = generate_client_profiles(num_clients_total, seed=42)

        def run_strategy(strategy_name, profiles, rng_seed=0):
            import numpy as np
            reset_energy(profiles)
            rng = np.random.default_rng(rng_seed)
            global_params = get_model_params(ECGNet())
            total_energy, total_trained = 0.0, 0

            for rnd in range(1, energy_rounds + 1):
                maybe_recharge(profiles, rnd, interval=recharge_interval)
                if strategy_name == "resource_aware":
                    train_ids, _ = select_clients_resource_aware(profiles, fraction=fraction)
                else:
                    train_ids, _ = select_clients_random(profiles, fraction=fraction, rng=rng)

                updates = []
                for cid in train_ids:
                    new_params, n, _ = clients[cid].fit(global_params, config={})
                    updates.append((new_params, n))
                    deplete(profiles[cid])
                    total_energy += profiles[cid].energy_per_round
                    total_trained += 1
                if updates:
                    global_params = weighted_average_params(updates)

            loss, auc = evaluate_on_dataset(global_params, X_val, y_val)
            return {"energy": total_energy, "avg_trained": total_trained / energy_rounds, "auc": auc, "loss": loss}

        with st.spinner("Running random-selection baseline..."):
            result_random = run_strategy("random", copy.deepcopy(profiles_template))
        with st.spinner("Running resource-aware selection..."):
            result_aware = run_strategy("resource_aware", copy.deepcopy(profiles_template))

        st.session_state.energy_results = (result_random, result_aware)

    if "energy_results" in st.session_state:
        result_random, result_aware = st.session_state.energy_results
        df = pd.DataFrame({
            "Random": [result_random["energy"], result_random["avg_trained"], result_random["auc"]],
            "Resource-Aware": [result_aware["energy"], result_aware["avg_trained"], result_aware["auc"]],
        }, index=["Total energy spent", "Avg clients trained/round", "Final val AUC"])
        st.table(df)
        st.bar_chart(pd.DataFrame({
            "Total energy": [result_random["energy"], result_aware["energy"]],
        }, index=["Random", "Resource-Aware"]))
