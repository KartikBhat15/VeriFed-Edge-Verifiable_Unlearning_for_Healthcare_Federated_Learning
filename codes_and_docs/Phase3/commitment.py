"""
Cryptographic commitments over model parameters, using a Merkle tree.

A commitment lets any party later verify "this exact set of parameters existed
at this point" without needing to store or transmit the full model — only the
root hash needs to be published/recorded. Before and after an unlearning
operation, we commit the model; the pair (pre_commitment, post_commitment)
becomes part of the signed verification certificate (see certificate.py).
"""

import hashlib
import numpy as np


def _hash_layer(layer: np.ndarray) -> bytes:
    """SHA-256 hash of one parameter tensor's raw bytes (plus its shape, so
    two differently-shaped-but-identically-valued layers don't collide)."""
    h = hashlib.sha256()
    h.update(str(layer.shape).encode())
    h.update(np.ascontiguousarray(layer).tobytes())
    return h.digest()


def _combine(left: bytes, right: bytes) -> bytes:
    h = hashlib.sha256()
    h.update(left)
    h.update(right)
    return h.digest()


def build_merkle_root(params) -> str:
    """
    params: list of numpy arrays (one per model layer, e.g. from
    get_model_params() in model.py).

    Returns the Merkle root as a hex string. This is the "commitment" — a
    single fixed-size fingerprint of the entire parameter set. Changing even
    one weight anywhere in the model changes the root completely.
    """
    if not params:
        raise ValueError("build_merkle_root: params list is empty")

    leaves = [_hash_layer(p) for p in params]

    level = leaves
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                next_level.append(_combine(level[i], level[i + 1]))
            else:
                # odd one out: promote unchanged to the next level
                next_level.append(level[i])
        level = next_level

    return level[0].hex()


def commitment_summary(params, label: str = "") -> str:
    """Human-readable one-line summary, useful for logging/certificates."""
    root = build_merkle_root(params)
    return f"{label + ': ' if label else ''}{root[:16]}...{root[-8:]}"
