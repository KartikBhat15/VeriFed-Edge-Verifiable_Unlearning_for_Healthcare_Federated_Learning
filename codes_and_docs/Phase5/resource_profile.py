"""
Simulated client resource profiles for adaptive, energy-aware FL.

We don't have real hospital/wearable hardware telemetry, so each client gets
a fixed, randomly-assigned "device class" at creation -- a compute_score
(how capable the device is, analogous to a wearable vs. a hospital-grade
ECG monitor) and a max_energy budget. This class doesn't change over time
(a device's hardware doesn't upgrade itself mid-deployment), but its
CURRENT energy does: training costs energy each round it participates, and
it periodically recharges (simulating overnight charging), which is what
gives the "adaptive train / inference-only mode switch" something real to
react to -- a client that's run its battery down switches to inference-only
until it's recharged enough to train again.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class ClientProfile:
    client_id: int
    compute_score: float   # 0.3 (weak wearable) .. 1.0 (hospital-grade server), fixed for the client's lifetime
    max_energy: float      # battery capacity, fixed
    current_energy: float  # depletes with training, replenished by recharge
    energy_per_round: float  # cost to train one round -- weaker devices spend more per round (less efficient)


def generate_client_profiles(num_clients: int, seed: int = 0):
    """
    Assigns each client a fixed device class. Weaker devices (lower
    compute_score) get a higher energy_per_round cost, reflecting that
    less efficient hardware burns more energy to do the same training work
    -- e.g. a wearable's ECG chip vs. a hospital cluster server.
    """
    rng = np.random.default_rng(seed)
    profiles = []
    for cid in range(num_clients):
        compute_score = float(rng.uniform(0.3, 1.0))
        max_energy = 100.0
        # inverse relationship: weaker devices cost more energy per round
        energy_per_round = float(20.0 * (1.4 - compute_score))
        profiles.append(ClientProfile(
            client_id=cid,
            compute_score=compute_score,
            max_energy=max_energy,
            current_energy=max_energy,
            energy_per_round=energy_per_round,
        ))
    return profiles


def reset_energy(profiles):
    """Reset all clients to full battery -- used to give two different
    selection strategies a fair, identical starting point for comparison."""
    for p in profiles:
        p.current_energy = p.max_energy


def can_train(profile: ClientProfile) -> bool:
    return profile.current_energy >= profile.energy_per_round


def deplete(profile: ClientProfile):
    profile.current_energy = max(0.0, profile.current_energy - profile.energy_per_round)


def maybe_recharge(profiles, round_num: int, interval: int = 3, amount_fraction: float = 0.4):
    """Every `interval` rounds, every client recharges by amount_fraction of
    its max capacity (simulating periodic/overnight charging), capped at
    max_energy."""
    if round_num % interval != 0:
        return
    for p in profiles:
        p.current_energy = min(p.max_energy, p.current_energy + amount_fraction * p.max_energy)


def select_clients_resource_aware(profiles, fraction: float = 0.5):
    """
    Adaptive, energy-aware selection: among clients that CAN currently
    afford to train, rank by a resource score (compute capability weighted
    by remaining energy fraction) and select the top `fraction`. Clients
    that can't afford to train, or weren't selected, run in
    inference-only mode this round.

    Returns (train_ids, inference_only_ids).
    """
    eligible = [p for p in profiles if can_train(p)]
    ineligible = [p for p in profiles if not can_train(p)]

    def resource_score(p):
        return p.compute_score * (p.current_energy / p.max_energy)

    eligible_sorted = sorted(eligible, key=resource_score, reverse=True)
    num_to_select = max(1, int(round(fraction * len(profiles))))
    selected = eligible_sorted[:num_to_select]
    not_selected = eligible_sorted[num_to_select:]

    train_ids = [p.client_id for p in selected]
    inference_only_ids = [p.client_id for p in not_selected] + [p.client_id for p in ineligible]
    return train_ids, inference_only_ids


def select_clients_random(profiles, fraction: float = 0.5, rng=None):
    """
    Baseline for comparison: randomly select clients regardless of resource
    score. Still respects the hard constraint that a client with
    insufficient energy cannot train (that's a physical limit, not a policy
    choice) -- the comparison is about whether SMART selection among the
    eligible clients beats RANDOM selection among them, not about ignoring
    energy limits altogether.
    """
    if rng is None:
        rng = np.random.default_rng()
    eligible = [p for p in profiles if can_train(p)]
    ineligible = [p for p in profiles if not can_train(p)]

    num_to_select = max(1, int(round(fraction * len(profiles))))
    num_to_select = min(num_to_select, len(eligible)) if eligible else 0

    if num_to_select > 0:
        selected_ids = rng.choice([p.client_id for p in eligible], size=num_to_select, replace=False)
        selected_ids = set(selected_ids.tolist())
    else:
        selected_ids = set()

    train_ids = [p.client_id for p in eligible if p.client_id in selected_ids]
    inference_only_ids = [p.client_id for p in eligible if p.client_id not in selected_ids] + \
                          [p.client_id for p in ineligible]
    return train_ids, inference_only_ids
