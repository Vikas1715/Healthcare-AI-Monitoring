"""
markov_model.py
---------------
Markov Chain model for patient health-state transitions.

States:  0 = Healthy  |  1 = At Risk  |  2 = Critical

Functions exported (imported by main.py):
  - get_markov_model(df)
  - simulate_next_state(state_name, matrix)
  - multi_step_simulation(state_idx, matrix, steps)
"""

import numpy as np
import pandas as pd

STATE_NAMES = ["Healthy", "At Risk", "Critical"]


def _assign_state(row: pd.Series) -> int:
    """
    Assign a health state to a patient row.
      Healthy  – ejection_fraction >= 50 AND serum_creatinine <= 1.2
      Critical – ejection_fraction <  35 OR  serum_creatinine >  2.0
      At Risk  – everything else
    """
    ef = row["ejection_fraction"]
    sc = row["serum_creatinine"]
    if ef < 35 or sc > 2.0:
        return 2   # Critical
    elif ef >= 50 and sc <= 1.2:
        return 0   # Healthy
    else:
        return 1   # At Risk


def get_markov_model(df: pd.DataFrame) -> dict:
    """
    Build the 3×3 transition matrix from the dataset and return all
    data needed by the frontend (matrix, state distribution, state counts).
    """
    df = df.copy()
    df["state"] = df.apply(_assign_state, axis=1)

    # Sort by 'time' and treat consecutive records as transitions
    df_sorted = df.sort_values("time").reset_index(drop=True)
    states    = df_sorted["state"].values

    # Count raw transitions
    counts = np.zeros((3, 3), dtype=float)
    for i in range(len(states) - 1):
        counts[states[i], states[i + 1]] += 1

    # Add Laplace smoothing so no row is all-zero
    counts += 0.1

    # Normalise rows → probabilities
    row_sums   = counts.sum(axis=1, keepdims=True)
    trans_mat  = (counts / row_sums).round(4)

    # State distribution
    state_counts = np.bincount(states, minlength=3)
    state_dist   = (state_counts / state_counts.sum()).round(4)

    return {
        "matrix":      trans_mat.tolist(),
        "state_names": STATE_NAMES,
        "state_distribution": {
            STATE_NAMES[i]: float(state_dist[i]) for i in range(3)
        },
        "state_counts": {
            STATE_NAMES[i]: int(state_counts[i]) for i in range(3)
        },
    }


def simulate_next_state(state_name: str, matrix: list) -> dict:
    """
    One-step prediction: given current state name, sample next state.
    """
    state_idx = STATE_NAMES.index(state_name) if state_name in STATE_NAMES else 0
    probs     = matrix[state_idx]
    next_idx  = int(np.random.choice(3, p=probs))

    return {
        "current_state": STATE_NAMES[state_idx],
        "next_state":    STATE_NAMES[next_idx],
        "probabilities": {STATE_NAMES[i]: round(probs[i], 4) for i in range(3)},
    }


def multi_step_simulation(start_idx: int,
                           matrix: list,
                           steps: int = 10) -> dict:
    """
    Simulate a patient trajectory over `steps` time steps.
    """
    T          = np.array(matrix)
    trajectory = [start_idx]
    current    = start_idx
    for _ in range(steps):
        current = int(np.random.choice(3, p=T[current]))
        trajectory.append(current)

    return {
        "steps":      steps,
        "trajectory": [STATE_NAMES[s] for s in trajectory],
        "indices":    trajectory,
    }
