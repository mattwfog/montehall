"""Jersey identity as a track-level inference: lineup-constrained decoding and evidence aggregation.

A single crop is a weak reading; a track is many crops, a few of them legible.
Given per-frame distributions over 100 numbers plus `illegible` (from
`candidates.jersey_classifier`), this module (1) restricts each distribution to
the team's lineup, the numbers actually on the pitch, and (2) accumulates
legibility-weighted log-likelihoods over the track, deciding only when the
posterior is decisive and enough legible evidence has been seen. Both steps are
pure functions so the decision rule is testable without a model.
"""

from __future__ import annotations

import numpy as np

NUM_NUMBERS = 100
ILLEGIBLE = NUM_NUMBERS
DEFAULT_DECISION = {
    "min_posterior": 0.9,
    "min_evidence": 1.0,  # weighted frame equivalents before a decision
    "min_frame_legibility": 0.2,  # frames below this carry no number information
    "legibility_power": 2.0,  # weight = legibility ** power; near-illegible frames barely count
    "max_frame_nats": 3.0,  # a single frame cannot move any candidate by more than this
    "log_floor": 1e-4,
}


def restrict_to_lineup(probabilities, lineup):
    """Renormalise the number mass over `lineup`, keeping the illegible mass unchanged.

    Returns a (101,) distribution with zero mass on numbers outside the lineup.
    An empty or None lineup leaves the distribution unchanged.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (NUM_NUMBERS + 1,):
        raise ValueError(f"Expected a {NUM_NUMBERS + 1}-way distribution, got {p.shape}")
    if not lineup:
        return p.copy()
    numbers = sorted({int(n) for n in lineup})
    if any(not 0 <= n < NUM_NUMBERS for n in numbers):
        raise ValueError(f"Lineup numbers must be in 0..{NUM_NUMBERS - 1}: {numbers}")
    out = np.zeros_like(p)
    out[ILLEGIBLE] = p[ILLEGIBLE]
    mass = p[numbers].sum()
    legible = 1.0 - p[ILLEGIBLE]
    if mass > 0:
        out[numbers] = p[numbers] / mass * legible
    else:
        out[numbers] = legible / len(numbers)
    return out


def aggregate_track(distributions, lineup=None, decision=DEFAULT_DECISION):
    """Posterior over numbers for one track from its per-frame distributions.

    Each frame contributes legibility ** power (legibility = 1 - p_illegible) times
    the log of its lineup-restricted number distribution, and frames under
    `min_frame_legibility` contribute nothing: a sharp number peak on an unreadable
    crop is noise, not a vote, and forty such frames must not outweigh two legible
    ones. Each frame's log-likelihoods are clipped to `max_frame_nats` below the
    frame's best candidate, so a run of confidently wrong frames cannot drive the
    posterior to certainty on its own: certainty has to come from agreement across
    frames. The decision needs the posterior to reach `min_posterior` and the
    summed weights to reach `min_evidence`.
    """
    frames = np.asarray(distributions, dtype=float).reshape(-1, NUM_NUMBERS + 1)
    candidates = sorted({int(n) for n in lineup}) if lineup else list(range(NUM_NUMBERS))
    score = np.zeros(len(candidates))
    evidence = 0.0
    for p in frames:
        restricted = restrict_to_lineup(p, lineup)
        legibility = float(1.0 - restricted[ILLEGIBLE])
        if legibility < decision["min_frame_legibility"]:
            continue
        weight = legibility ** decision["legibility_power"]
        numbers = restricted[candidates]
        numbers = (
            numbers / numbers.sum()
            if numbers.sum() > 0
            else np.full(len(candidates), 1.0 / len(candidates))
        )
        log_likelihood = np.log(np.maximum(numbers, decision["log_floor"]))
        score += weight * np.maximum(
            log_likelihood - log_likelihood.max(), -decision["max_frame_nats"]
        )
        evidence += weight
    posterior = np.exp(score - score.max())
    posterior /= posterior.sum()
    best = int(np.argmax(posterior))
    decided = bool(
        posterior[best] >= decision["min_posterior"] and evidence >= decision["min_evidence"]
    )
    return {
        "number": candidates[best] if decided else None,
        "top_number": candidates[best],
        "posterior": float(posterior[best]),
        "runner_up_posterior": float(np.sort(posterior)[-2]) if len(posterior) > 1 else 0.0,
        "evidence": float(evidence),
        "frames": len(frames),
        "status": "identified"
        if decided
        else "insufficient_evidence"
        if evidence < decision["min_evidence"]
        else "undecided",
    }


def identify_tracks(track_distributions, lineups, decision=DEFAULT_DECISION):
    """{track_key: [distributions]} with {track_key: lineup or None} → {track_key: decision}."""
    return {
        key: aggregate_track(frames, lineups.get(key), decision)
        for key, frames in track_distributions.items()
    }
