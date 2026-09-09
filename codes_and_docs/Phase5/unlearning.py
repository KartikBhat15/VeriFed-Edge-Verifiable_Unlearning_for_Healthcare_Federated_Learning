"""
Gradient-based approximate unlearning.

Rather than retraining from scratch without the target client's data (which
is what "exact" unlearning would require, and which is computationally
prohibitive at hospital-network scale), we take a small number of gradient
*ascent* steps on the target client's data -- directly increasing the loss
the model has on that data, which pushes the model away from having fit it.
This is the practical, efficient interpretation of the SISA-style philosophy
we cited (Bourtoule et al., 2021): isolate and limit the blast radius of the
removal operation. To prevent the ascent step from degrading the model's
general performance (catastrophic forgetting well beyond the target data),
we follow ascent with a short stabilization phase of ordinary training on a
retain set (data NOT being forgotten).
"""

import copy
import torch
import torch.nn as nn

from model import get_model_params, set_model_params

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gradient_ascent_unlearn(
    model_class,
    current_params,
    forget_X,
    forget_y,
    retain_X=None,
    retain_y=None,
    ascent_epochs: int = 3,
    ascent_lr: float = 1e-3,
    stabilize_epochs: int = 2,
    stabilize_lr: float = 1e-3,
    max_grad_norm: float = 1.0,
):
    """
    Returns new model parameters with the target (forget) client's influence
    approximately removed.

    model_class: the model constructor (e.g. ECGNet), used to build a fresh
        instance to operate on so we never mutate a model the caller still
        holds a reference to.
    current_params: the model parameters to unlearn FROM (e.g. the current
        global model, however it was produced -- flat or hierarchical FedAvg).
    forget_X, forget_y: the target client's data (the data being "deleted").
    retain_X, retain_y: optional data NOT being forgotten, used to stabilize
        the model after the ascent step so it doesn't collapse generally.
    """
    model = model_class().to(DEVICE)
    set_model_params(model, current_params)
    loss_fn = nn.BCEWithLogitsLoss()

    # ---------- Phase 1: gradient ascent on the forget set ----------
    opt = torch.optim.SGD(model.parameters(), lr=ascent_lr)
    X_f = torch.tensor(forget_X, dtype=torch.float32).to(DEVICE)
    y_f = torch.tensor(forget_y, dtype=torch.float32).to(DEVICE)

    model.train()
    for _ in range(ascent_epochs):
        opt.zero_grad()
        out = model(X_f)
        loss = loss_fn(out, y_f)
        (-loss).backward()  # ascent: maximize loss on the forget set
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()

    # ---------- Phase 2: stabilize on the retain set, if provided ----------
    if retain_X is not None and len(retain_X) > 0 and stabilize_epochs > 0:
        opt = torch.optim.Adam(model.parameters(), lr=stabilize_lr)
        X_r = torch.tensor(retain_X, dtype=torch.float32).to(DEVICE)
        y_r = torch.tensor(retain_y, dtype=torch.float32).to(DEVICE)
        for _ in range(stabilize_epochs):
            opt.zero_grad()
            out = model(X_r)
            loss = loss_fn(out, y_r)
            loss.backward()
            opt.step()

    return get_model_params(model)
