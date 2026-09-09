# Phase 5: Streamlit Dashboard

**Status: built, syntax-validated, headless-boot-validated, and every tab's
underlying logic validated against synthetic data.** Full click-through
testing wasn't possible in my environment (no browser automation available
here), so treat the first real run on your machine as the final check.

## What it is
`dashboard.py` — a single-file Streamlit app with three tabs, all reusing
the exact same library modules (`data_utils.py`, `model.py`, `client.py`,
`aggregation.py`, `commitment.py`, `mia.py`, `unlearning.py`,
`certificate.py`, `resource_profile.py`) as the command-line scripts from
Phases 1-4. This file contains no new FL/crypto logic of its own — it's UI
glue code calling into everything already validated.

- **Tab 1 (Training):** live-updating loss/AUC chart while flat FedAvg
  trains on real PTB-XL data.
- **Tab 2 (Unlearning & Verification):** pick a client, submit a deletion
  request, watch the full pipeline run — commitment before/after, MIA
  before/after as metrics with a delta indicator, the full certificate as
  JSON, and a "tamper test" button that corrupts the certificate and proves
  verification catches it.
- **Tab 3 (Energy-Aware Selection):** run random vs. resource-aware client
  selection side by side, see total energy spent and final accuracy in a
  table plus a bar chart.

## How to run

```powershell
streamlit run dashboard.py
```

This opens in your browser automatically (usually `http://localhost:8501`).
In the sidebar: enter your PTB-XL root path, pick number of clients and
alpha, click **"Load PTB-XL and partition clients"** — this has to be done
once before any tab does anything. Then use each tab independently; they
share the same loaded dataset and (once trained) the same model.

## How I validated this without click-testing

1. **Syntax check** (`py_compile`) — passed clean.
2. **Headless server boot** — launched the actual app with
   `streamlit run --server.headless true`, confirmed it serves HTTP 200 with
   no Python tracebacks in the startup log.
3. **Logic extraction test** — pulled the exact training loop, deletion
   request pipeline, and energy comparison logic out of the dashboard code
   and ran each one directly against synthetic data outside the UI layer,
   confirming: training loop runs without error, commitment actually
   changes after unlearning, certificate signs and verifies, the tamper
   test correctly gets rejected, and the energy comparison produces
   different real numbers for each strategy.

What this does NOT prove: that every button, slider, and `st.session_state`
interaction behaves correctly when you actually click through it in a
browser — Streamlit's rerun-on-every-interaction model has its own
failure modes (stale state, widgets not clearing) that only show up in
real use. If something looks visually wrong or a button doesn't do what
you expect, tell me exactly what happened and I'll fix it — that's a much
faster path than me guessing at UI behavior I can't observe directly.

## Known limitation
Training in Tab 1 runs synchronously in the Streamlit thread — for a large
`--num_rounds`, the browser tab will show a spinner and the page won't be
interactive until it finishes. This matches every other phase's design (no
background job queue), just worth knowing before you kick off a long run.
