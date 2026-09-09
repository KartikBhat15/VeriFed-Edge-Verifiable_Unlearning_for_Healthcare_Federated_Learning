"""
Manual FedAvg weighted-averaging helper.

Flower's built-in FedAvg strategy operates at a single level (server <-> clients).
To get a genuine 3-tier hierarchy (edge -> cluster -> global), each "cluster" in
our simulation is itself a Flower client that runs its OWN inner FedAvg loop
across its edge devices before handing a single aggregated update up to the
outer (global) Flower server. This module provides that inner aggregation step.
"""

import numpy as np


def weighted_average_params(updates):
    """
    updates: list of (params, num_examples) tuples, where params is a list of
    numpy arrays (one per model layer), all with matching shapes across updates.

    Returns a single list of numpy arrays: the sample-weighted average of all
    updates, i.e. standard FedAvg aggregation.
    """
    total_examples = sum(num_examples for _, num_examples in updates)
    if total_examples == 0:
        raise ValueError("weighted_average_params: total_examples is 0")

    num_layers = len(updates[0][0])
    original_dtypes = [layer.dtype for layer in updates[0][0]]
    aggregated = [np.zeros(layer.shape, dtype=np.float64) for layer in updates[0][0]]

    for params, num_examples in updates:
        weight = num_examples / total_examples
        for i in range(num_layers):
            aggregated[i] += params[i].astype(np.float64) * weight

    # cast each layer back to its original dtype (int buffers like BatchNorm's
    # num_batches_tracked round to the nearest int rather than truncate)
    result = []
    for layer, dtype in zip(aggregated, original_dtypes):
        if np.issubdtype(dtype, np.integer):
            result.append(np.round(layer).astype(dtype))
        else:
            result.append(layer.astype(np.float32))
    return result
