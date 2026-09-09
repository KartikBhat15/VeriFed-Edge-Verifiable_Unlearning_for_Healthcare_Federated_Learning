# Phase 4: Adaptive, Energy-Aware Client Selection

**Status: built and validated on synthetic data.**

## Files

- `resource_profile.py` — simulated per-client resource profiles. Since we
  don't have real hospital hardware telemetry, each client gets a fixed
  `compute_score` (0.3-1.0, a device-class rating that doesn't change) and a
  `max_energy` budget whose *current* level does change: training depletes
  it, and `maybe_recharge()` periodically restores some of it (simulating
  overnight charging). Weaker devices (lower `compute_score`) cost more
  energy per round — the inverse relationship reflects that less efficient
  hardware burns more energy to do the same training work.

  Two selection strategies are implemented:
  - `select_clients_resource_aware(...)` — ranks eligible clients by
    `compute_score × remaining_energy_fraction`, selects the top fraction.
  - `select_clients_random(...)` — the baseline: picks randomly among
    clients that can currently afford to train (still respects the hard
    physical constraint that a client with insufficient energy can't train
    — the comparison is about smart vs. random selection *among the
    eligible*, not about ignoring energy limits).

- `simulate_energy_aware.py` — runs the *same* FedAvg training twice, from
  identical starting conditions, once with each strategy, then reports a
  side-by-side comparison: total energy spent, average clients trained per
  round, and final AUC on PTB-XL's official validation fold.

## How to run

```
python simulate_energy_aware.py --ptbxl_root <path> --num_clients 8 --num_rounds 15 --fraction 0.5 --recharge_interval 3
```

## An honest, two-sided result — read before reporting a single number

I ran this under two different parameter regimes while validating it, and
**the direction of the result flips** depending on how much energy pressure
the simulation is under:

**Moderate pressure** (`--fraction 0.5 --recharge_interval 3`, default-ish):
resource-aware selection used **~19% less total energy** for equal-or-better
accuracy. This is the clean "energy savings" story your synopsis promises.

**Heavy, sustained pressure** (`--fraction 0.7 --recharge_interval 8`, long
run): resource-aware selection used **slightly *more* total energy**, but
sustained meaningfully more client participation (fewer clients got
excluded from training due to exhaustion — 5.85 vs 5.50 average clients
trained per round) and achieved better final accuracy. Per training event,
resource-aware selection was still measurably more energy-efficient
(11.64 vs 11.88 units per client-round) — it just ran more total training
events because it kept more clients healthy and eligible over time, rather
than letting them all drain simultaneously the way random selection does.

**Both outcomes are correct and explainable, not a bug** — they're the same
underlying mechanism (favor efficient, well-resourced clients) producing
different trade-offs depending on how scarce energy is relative to the
recharge rate. For your report, I'd lead with the moderate-pressure result
(matches the synopsis's stated expected outcome directly), but if a guide or
examiner pushes on "does it always save energy," the honest answer is "it
depends on the energy-scarcity regime, and here's why" — which is a
stronger answer than pretending it's a strict win in every case.

## Tuning knobs
- `--fraction` — what fraction of clients train each round
- `--recharge_interval` — lower = more frequent recharging = less energy
  pressure = more likely to see the "clean savings" result
- Client resource profiles are regenerated with a fixed seed (42) each run,
  so results are reproducible round to round

## Next steps (Phase 5, Weeks 10-11)
Streamlit dashboard — triggering training/deletion requests and viewing
certificates and training/energy metrics through a UI instead of the
command line.
