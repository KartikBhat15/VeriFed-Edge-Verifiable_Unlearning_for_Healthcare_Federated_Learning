"""
Membership inference attack (MIA), used defensively here as a verification
signal rather than offensively as in the original literature (Shokri et al.,
2017). The idea: if a client's data was genuinely "forgotten," the model's
per-sample loss on that client's data should no longer be distinguishable
from its loss on data it never saw. We use the standard loss-threshold
attack (Yeom et al., 2018) as a simple, well-understood baseline: an attacker
who has seen both a member and a non-member loss distribution picks the
threshold that best separates them, then reports how accurately that
threshold classifies held-out samples from each group.

A high MIA success rate (well above 0.5 / random guessing) before unlearning,
dropping toward 0.5 after unlearning, is the empirical evidence that the
target client's influence was actually removed -- not just that accuracy
changed.
"""

import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def per_sample_loss(model, X, y):
    """Per-sample BCE loss (no reduction), used as the MIA attack signal."""
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        y_t = torch.tensor(y, dtype=torch.float32).to(DEVICE)
        out = model(X_t)
        losses = loss_fn(out, y_t).mean(dim=1)  # mean across the 5 label dims
    return losses.cpu().numpy()


def mia_success_rate(model, member_X, member_y, nonmember_X, nonmember_y):
    """
    Loss-threshold membership inference attack.

    Lower loss => model fits this sample well => attacker guesses "member".
    We pick the threshold (via a simple sweep) that maximizes separation
    between the two loss distributions, then report the resulting attack
    accuracy: fraction of members correctly called "member" plus fraction of
    non-members correctly called "non-member", averaged.

    Returns a float in [0, 1]. ~0.5 = random guessing (no detectable
    membership signal, i.e. consistent with successful unlearning).
    Meaningfully > 0.5 = the model still "remembers" this data.
    """
    member_losses = per_sample_loss(model, member_X, member_y)
    nonmember_losses = per_sample_loss(model, nonmember_X, nonmember_y)

    if len(member_losses) == 0 or len(nonmember_losses) == 0:
        return 0.5  # no data to attack with; report uninformative baseline

    candidate_thresholds = np.unique(
        np.concatenate([member_losses, nonmember_losses])
    )

    best_acc = 0.0
    for t in candidate_thresholds:
        # attacker calls "member" when loss < t
        member_correct = np.mean(member_losses < t)
        nonmember_correct = np.mean(nonmember_losses >= t)
        acc = (member_correct + nonmember_correct) / 2.0
        if acc > best_acc:
            best_acc = acc

    return float(best_acc)
