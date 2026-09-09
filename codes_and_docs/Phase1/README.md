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
