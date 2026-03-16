"""
main.py
-------
Flask backend for the Healthcare AI Monitoring application.

Run with:
    python main.py
Server starts on http://localhost:8000
"""

import io
import json

import numpy as np
import pandas as pd
from flask import Flask, request
from flask_cors import CORS

from analysis     import (explore_dataset, marginal_probability,
                           joint_probability, conditional_probability,
                           mle_estimation, kl_divergence,
                           generate_synthetic_data)
from markov_model import (get_markov_model, simulate_next_state,
                           multi_step_simulation)
from hmm_model    import run_hmm

# ── App setup ──────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# In-memory dataset store (single-session)
_df_store = {}
SESSION_KEY = "current"


# ── JSON helper (handles numpy types) ─────────────────────────────
class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def _json(data, status=200):
    return app.response_class(
        response=json.dumps(data, cls=_NumpyEncoder),
        status=status,
        mimetype="application/json",
    )


def _get_df() -> pd.DataFrame:
    if SESSION_KEY not in _df_store:
        raise ValueError("No dataset loaded. Please upload a CSV first.")
    return _df_store[SESSION_KEY]


# ══════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════
@app.route("/")
@app.route("/health")
def health():
    return _json({"status": "ok",
                  "message": "Healthcare AI Monitoring API is running"})


# ══════════════════════════════════════════════════════════════
# 1. UPLOAD DATASET
# ══════════════════════════════════════════════════════════════
@app.route("/upload", methods=["POST"])
def upload_dataset():
    if "file" not in request.files:
        return _json({"detail": "No file part in request."}, 400)

    file = request.files["file"]
    if not file.filename.lower().endswith(".csv"):
        return _json({"detail": "Only CSV files are accepted."}, 400)

    try:
        df = pd.read_csv(io.BytesIO(file.read()))
    except Exception as exc:
        return _json({"detail": f"Could not parse CSV: {exc}"}, 400)

    required = {"age", "DEATH_EVENT", "ejection_fraction",
                "serum_creatinine", "diabetes", "high_blood_pressure"}
    missing  = required - set(df.columns)
    if missing:
        return _json({"detail": f"CSV is missing required columns: {missing}"}, 422)

    _df_store[SESSION_KEY] = df
    return _json(explore_dataset(df))


# ══════════════════════════════════════════════════════════════
# 2. MARGINAL PROBABILITY
# ══════════════════════════════════════════════════════════════
@app.route("/marginal")
def marginal():
    try:
        return _json(marginal_probability(_get_df()))
    except ValueError as exc:
        return _json({"detail": str(exc)}, 400)


# ══════════════════════════════════════════════════════════════
# 3. JOINT PROBABILITY
# ══════════════════════════════════════════════════════════════
@app.route("/joint")
def joint():
    try:
        age_threshold = int(request.args.get("age_threshold", 60))
        return _json(joint_probability(_get_df(), age_threshold))
    except ValueError as exc:
        return _json({"detail": str(exc)}, 400)


# ══════════════════════════════════════════════════════════════
# 4. CONDITIONAL PROBABILITY
# ══════════════════════════════════════════════════════════════
@app.route("/conditional")
def conditional():
    try:
        condition = request.args.get("condition", "age_gt_60")
        return _json(conditional_probability(_get_df(), condition))
    except ValueError as exc:
        return _json({"detail": str(exc)}, 400)


# ══════════════════════════════════════════════════════════════
# 5. MLE
# ══════════════════════════════════════════════════════════════
@app.route("/mle")
def mle():
    try:
        return _json(mle_estimation(_get_df()))
    except ValueError as exc:
        return _json({"detail": str(exc)}, 400)


# ══════════════════════════════════════════════════════════════
# 6. KL DIVERGENCE
# ══════════════════════════════════════════════════════════════
@app.route("/kl-divergence")
def kl():
    try:
        assumed_p = float(request.args.get("assumed_p", 0.30))
        if not (0 < assumed_p < 1):
            return _json({"detail": "assumed_p must be strictly between 0 and 1."}, 400)
        return _json(kl_divergence(_get_df(), assumed_p))
    except ValueError as exc:
        return _json({"detail": str(exc)}, 400)


# ══════════════════════════════════════════════════════════════
# 7. MARKOV CHAIN
# ══════════════════════════════════════════════════════════════
@app.route("/markov/matrix")
def markov_matrix():
    try:
        return _json(get_markov_model(_get_df()))
    except ValueError as exc:
        return _json({"detail": str(exc)}, 400)


@app.route("/markov/simulate", methods=["POST"])
def markov_simulate():
    try:
        body        = request.get_json(force=True) or {}
        state_idx   = int(body.get("current_state", 0))
        steps       = int(body.get("steps", 10))
        steps       = max(5, min(steps, 50))

        mdata       = get_markov_model(_get_df())
        matrix      = mdata["matrix"]
        names       = mdata["state_names"]
        state_name  = names[state_idx] if 0 <= state_idx < 3 else "Healthy"

        next_s      = simulate_next_state(state_name, matrix)
        trajectory  = multi_step_simulation(state_idx, matrix, steps)

        return _json({
            "next_state_prediction": next_s,
            "trajectory":            trajectory,
            "matrix":                matrix,
            "state_names":           names,
        })
    except ValueError as exc:
        return _json({"detail": str(exc)}, 400)


# ══════════════════════════════════════════════════════════════
# 8. HIDDEN MARKOV MODEL
# ══════════════════════════════════════════════════════════════
@app.route("/hmm/predict")
def hmm_predict():
    try:
        return _json(run_hmm(_get_df()))
    except ValueError as exc:
        return _json({"detail": str(exc)}, 400)


# ══════════════════════════════════════════════════════════════
# 9. SYNTHETIC DATA
# ══════════════════════════════════════════════════════════════
@app.route("/generative/synthetic")
def synthetic():
    try:
        n_samples = int(request.args.get("n_samples", 20))
        n_samples = max(10, min(n_samples, 200))
        return _json(generate_synthetic_data(_get_df(), n_samples))
    except ValueError as exc:
        return _json({"detail": str(exc)}, 400)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n🏥  Healthcare AI Monitoring API")
    print("   http://localhost:8000\n")
    app.run(host="0.0.0.0", port=8000, debug=True)
