# VeriFed-Edge — Week 1-2: Flat FedAvg Baseline on PTB-XL

This is your Week 1-2 deliverable: a working (tested) federated learning
pipeline on real ECG data, before we add the hierarchical layer (Week 3-4)
and unlearning (Week 5-7).

## Files
- `data_utils.py` — PTB-XL loading, label mapping (5 diagnostic superclasses), Dirichlet non-IID client partitioning
- `model.py` — 1D CNN for 12-lead ECG (`ECGNet`)
- `client.py` — Flower `NumPyClient` wrapping local train/eval
- `simulate.py` — runs the real experiment on PTB-XL
- `test_pipeline.py` — **run this first**, validates everything with synthetic data (no download needed, already verified working)
- `download_ptbxl.sh` — pulls PTB-XL from PhysioNet (~1.7GB)

## How to run on Colab

1. Upload all `.py` files to your Colab session (or `git clone` if you push this to a repo).
2. Install dependencies:
   ```
   !pip install "flwr[simulation]" torch wfdb scikit-learn -q
   ```
3. Sanity check first (no dataset needed, ~10 seconds):
   ```
   !python test_pipeline.py
   ```
4. Download PTB-XL:
   ```
   !bash download_ptbxl.sh
   ```
5. Run the real experiment:
   ```
   !python simulate.py --ptbxl_root ptbxl/physionet.org/files/ptb-xl/1.0.3 --num_clients 8 --num_rounds 15 --alpha 0.5
   ```

## What to expect
- PTB-XL benchmark papers report ~0.90-0.93 macro AUC on this 5-class task with centralized training; federated with non-IID splits will land somewhat lower — that gap itself is worth reporting.
- `--alpha` controls non-IID severity: 0.5 is realistic hospital heterogeneity, try 5.0 for a near-IID sanity check, 0.1 for extreme skew.
- Training on full PTB-XL (~18,000 train records) will take longer than the synthetic test — expect several minutes per round on Colab's free GPU depending on `--num_clients`.

## Next steps (Week 3-4)
Once this baseline gives you a stable AUC number, we'll restructure `simulate.py`
into the 3-tier hierarchy from your block diagram: cluster-level FedAvg first,
then a global aggregation step across clusters.

---

# Week 3-4: 3-Tier Hierarchical FedAvg

**Status: built and validated with synthetic data.**

## New files
- `aggregation.py` — manual weighted-FedAvg helper, used for the inner
  (cluster-level) aggregation step
- `hierarchical_client.py` — `ClusterClient`: a Flower client that is itself a
  mini federation. From the global server's point of view it's one ordinary
  client; internally, `fit()` distributes parameters to its own edge devices,
  trains each one locally, and aggregates them via `aggregation.py` before
  reporting the cluster-level result upward. This is what makes it a genuine
  3-tier hierarchy (Edge -> Cluster -> Global) rather than a flat one.
- `simulate_hierarchical.py` — the real experiment runner on PTB-XL
- `test_hierarchical_pipeline.py` — synthetic-data sanity check (already run
  and passing: loss decreased 0.678 -> 0.674 -> 0.671 over 3 rounds with zero
  fit failures across 3 clusters x 3 edges each)

## New function in `data_utils.py`
- `hierarchical_partition()` — two-level Dirichlet partitioning. First splits
  records across clusters (hospitals), non-IID by default (`cluster_alpha=0.5`,
  realistic hospital-to-hospital heterogeneity), then splits each cluster's
  records across its own edge devices (`edge_alpha=5.0` by default — devices
  within one hospital are assumed to see a more similar case mix than hospitals
  see relative to each other, but you can tune this).

## How to run on Colab
Same setup as Week 1-2 (`pip install "flwr[simulation]" torch wfdb scikit-learn`,
then `download_ptbxl.sh`), then:
```
!python test_hierarchical_pipeline.py       # sanity check first, ~10s, no dataset needed
!python simulate_hierarchical.py --ptbxl_root ptbxl/physionet.org/files/ptb-xl/1.0.3 \
    --num_clusters 3 --edges_per_cluster 3 --num_global_rounds 15
```

## A real bug we hit and fixed
The first version of `aggregation.py` used `np.zeros_like(layer)` to build the
accumulator, which preserves the source array's dtype. PyTorch's BatchNorm
layers include an `int64` buffer (`num_batches_tracked`), so accumulating a
float64-weighted update into an int64 array crashed with a casting error —
silently failing every single client's `fit()` call (Flower logged "0 results,
3 failures" but the simulation kept running and just returned the untrained
initial parameters every round, which is a dangerous kind of failure since
nothing crashes loudly). Fixed by always accumulating in float64 and casting
each layer back to its original dtype (rounding for integer buffers) at the
end. Worth remembering for Week 5-7 too, since the unlearning code will do
similar parameter-level arithmetic.

## Tuning knobs
- `--num_clusters` / `--edges_per_cluster` — matches your block diagram's 3x3
  by default; raise `edges_per_cluster` for a more convincing non-IID story
- `--cluster_alpha` — lower = hospitals see more different case mixes
- `--edge_alpha` — lower = devices within one hospital see more different case
  mixes (defaults higher/more-IID than cluster_alpha, tune if your report needs
  device-level heterogeneity too)

## Next steps (Week 5-7)
Verifiable unlearning: cryptographic commitments (Merkle tree) over the
parameter history this hierarchy now produces, plus MIA-based verification
when a deletion request comes in.
