"""
PTB-XL data utilities for VeriFed-Edge.

PTB-XL ground truth: https://physionet.org/content/ptb-xl/
Standard benchmark task (Strodthoff et al. 2020): multi-label classification
into 5 diagnostic superclasses: NORM, MI, STTC, CD, HYP.

Expected directory layout after downloading PTB-XL (see download_ptbxl.sh):
    <ptbxl_root>/ptbxl_database.csv
    <ptbxl_root>/scp_statements.csv
    <ptbxl_root>/records100/...   (100Hz signals, use these — records500 is 500Hz raw)
"""

import ast
import numpy as np
import pandas as pd
import wfdb
from sklearn.preprocessing import MultiLabelBinarizer

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_ptbxl_metadata(ptbxl_root: str) -> pd.DataFrame:
    """Load the PTB-XL database CSV and parse the scp_codes column."""
    df = pd.read_csv(f"{ptbxl_root}/ptbxl_database.csv", index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    return df


def load_scp_statements(ptbxl_root: str) -> pd.DataFrame:
    agg_df = pd.read_csv(f"{ptbxl_root}/scp_statements.csv", index_col=0)
    return agg_df[agg_df.diagnostic == 1]


def aggregate_diagnostic(scp_codes: dict, agg_df: pd.DataFrame) -> list:
    """Map raw SCP codes for one record to the 5 diagnostic superclasses."""
    classes = set()
    for code in scp_codes.keys():
        if code in agg_df.index:
            classes.add(agg_df.loc[code].diagnostic_class)
    return list(classes)


def build_labels(df: pd.DataFrame, agg_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["diagnostic_superclass"] = df.scp_codes.apply(
        lambda codes: aggregate_diagnostic(codes, agg_df)
    )
    # drop records with no mapped superclass
    df = df[df.diagnostic_superclass.apply(len) > 0]
    return df


def load_raw_signals(df: pd.DataFrame, ptbxl_root: str) -> np.ndarray:
    """Load 100Hz 12-lead signals for every record in df. Returns (N, leads, T)."""
    signals = []
    for f in df.filename_lr:  # filename_lr = 100Hz path column in ptbxl_database.csv
        sig, _ = wfdb.rdsamp(f"{ptbxl_root}/{f}")
        signals.append(sig.T)  # (T, leads) -> (leads, T)
    return np.array(signals, dtype=np.float32)


def get_ptbxl_arrays(ptbxl_root: str):
    """
    Full pipeline: metadata -> labels -> signals -> official PTB-XL train/test split.
    PTB-XL recommends fold 10 as test, folds 1-8 as train, fold 9 as validation.
    Returns X_train, y_train, X_test, y_test, mlb (label binarizer)
    """
    meta = load_ptbxl_metadata(ptbxl_root)
    agg_df = load_scp_statements(ptbxl_root)
    meta = build_labels(meta, agg_df)

    mlb = MultiLabelBinarizer(classes=SUPERCLASSES)
    y_all = mlb.fit_transform(meta.diagnostic_superclass)

    train_mask = meta.strat_fold <= 8
    val_mask = meta.strat_fold == 9
    test_mask = meta.strat_fold == 10

    X_train = load_raw_signals(meta[train_mask], ptbxl_root)
    X_val = load_raw_signals(meta[val_mask], ptbxl_root)
    X_test = load_raw_signals(meta[test_mask], ptbxl_root)

    y_train = y_all[train_mask.values]
    y_val = y_all[val_mask.values]
    y_test = y_all[test_mask.values]

    return X_train, y_train, X_val, y_val, X_test, y_test, mlb


def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float = 0.5, seed: int = 42):
    """
    Partition record indices across clients with non-IID label skew.
    Uses the first active label per record as the partitioning key
    (standard trick for multi-label Dirichlet partitioning).
    alpha: lower = more non-IID (hospitals see very different case mixes),
           higher = closer to IID. 0.5 is a common realistic default.
    """
    rng = np.random.default_rng(seed)
    primary_label = labels.argmax(axis=1)  # (N,)
    num_classes = labels.shape[1]
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = np.where(primary_label == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))
        # convert proportions to split points
        split_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, split_points)
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    return [np.array(idxs) for idxs in client_indices]
