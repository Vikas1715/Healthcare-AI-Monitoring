"""
hmm_model.py
------------
Hidden Markov Model for healthcare monitoring.

Hidden States  (unobservable): Healthy | At Risk | Critical
Observed Syms  (observable)  : 0=Normal | 1=Mild | 2=Severe Symptoms

Functions exported (imported by main.py):
  - run_hmm(df)   → full HMM results dict
"""

import numpy as np
import pandas as pd

try:
    from hmmlearn import hmm as hmmlearn_hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

STATE_NAMES = ["Healthy", "At Risk", "Critical"]
OBS_NAMES   = ["Normal Symptoms", "Mild Symptoms", "Severe Symptoms"]

# ── Fixed illustrative parameters ──────────────────────────────────
START_PROBS = np.array([0.60, 0.30, 0.10])

TRANS_MATRIX = np.array([
    [0.70, 0.20, 0.10],
    [0.15, 0.65, 0.20],
    [0.05, 0.25, 0.70],
])

# P(observation | hidden_state)   rows=hidden, cols=observation
#                  Normal  Mild  Severe
EMISSION_MATRIX = np.array([
    [0.70, 0.20, 0.10],   # Healthy
    [0.20, 0.50, 0.30],   # At Risk
    [0.05, 0.25, 0.70],   # Critical
])


def _discretise(df: pd.DataFrame) -> np.ndarray:
    """Map clinical features → observation symbols {0, 1, 2}."""
    obs = []
    for _, row in df.iterrows():
        ef  = row["ejection_fraction"]
        sc  = row["serum_creatinine"]
        age = row["age"]
        if ef < 35 or sc > 2.0 or age >= 75:
            obs.append(2)   # Severe
        elif ef >= 50 and sc <= 1.2 and age < 60:
            obs.append(0)   # Normal
        else:
            obs.append(1)   # Mild
    return np.array(obs)


def _heuristic_decode(obs_seq: np.ndarray) -> np.ndarray:
    """MAP heuristic: choose hidden state most likely to emit each observation."""
    return np.array([int(np.argmax(EMISSION_MATRIX[:, o])) for o in obs_seq])


def run_hmm(df: pd.DataFrame) -> dict:
    """
    Decode the most-likely sequence of hidden health states using
    the Viterbi algorithm (hmmlearn) or a heuristic fallback.
    Returns everything the frontend needs to render the HMM section.
    """
    obs_seq = _discretise(df)
    n       = len(obs_seq)

    if HMM_AVAILABLE:
        try:
            model = hmmlearn_hmm.CategoricalHMM(n_components=3,
                                                n_iter=50,
                                                random_state=42)
            model.startprob_    = START_PROBS
            model.transmat_     = TRANS_MATRIX
            model.emissionprob_ = EMISSION_MATRIX
            X = obs_seq.reshape(-1, 1)
            _, hidden_seq = model.decode(X, algorithm="viterbi")
            method = "hmmlearn CategoricalHMM — Viterbi decoding"
        except Exception as exc:
            hidden_seq = _heuristic_decode(obs_seq)
            method = f"Heuristic fallback (hmmlearn error: {exc})"
    else:
        hidden_seq = _heuristic_decode(obs_seq)
        method = "Heuristic MAP decoder (hmmlearn not installed)"

    state_counts = np.bincount(hidden_seq, minlength=3)

    # Build per-patient sample for the table (first 20 rows)
    sample = [
        {
            "index":        int(i),
            "observation":  OBS_NAMES[obs_seq[i]],
            "hidden_state": STATE_NAMES[hidden_seq[i]],
        }
        for i in range(min(20, n))
    ]

    return {
        "method":             method,
        "sample_predictions": sample,
        "state_frequencies": {
            STATE_NAMES[i]: int(state_counts[i]) for i in range(3)
        },
        "total_records":   n,
        "emission_matrix": EMISSION_MATRIX.tolist(),
        "hidden_states":   STATE_NAMES,
        "obs_names":       OBS_NAMES,
        "trans_matrix":    TRANS_MATRIX.tolist(),
        "start_probs":     START_PROBS.tolist(),
    }
