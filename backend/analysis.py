"""
analysis.py
-----------
All probabilistic analysis functions for Healthcare AI Monitoring.

Functions exported (imported by main.py):
  - explore_dataset
  - marginal_probability
  - joint_probability
  - conditional_probability
  - mle_estimation
  - kl_divergence
  - generate_synthetic_data
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# 1. DATASET EXPLORATION
# ─────────────────────────────────────────────────────────────

def explore_dataset(df: pd.DataFrame) -> dict:
    """
    Return summary statistics, correlation matrix, scatter data,
    and a preview of the first 10 rows.
    """
    total   = len(df)
    deaths  = int(df["DEATH_EVENT"].sum())
    alive   = total - deaths

    # Per-column descriptive stats (only numeric columns)
    desc = df.describe(include=[float, int]).round(4)
    stats = {}
    for col in desc.columns:
        stats[col] = {
            "mean": float(desc.loc["mean", col]),
            "std":  float(desc.loc["std",  col]),
            "min":  float(desc.loc["min",  col]),
            "max":  float(desc.loc["max",  col]),
            "25%":  float(desc.loc["25%",  col]),
            "50%":  float(desc.loc["50%",  col]),
            "75%":  float(desc.loc["75%",  col]),
        }

    # Correlation matrix
    corr = df.corr(numeric_only=True).round(4)
    correlation = {
        "labels": list(corr.columns),
        "matrix": corr.values.tolist()
    }

    # Scatter data for key features
    scatter = {}
    for feat in ["age", "ejection_fraction", "serum_creatinine"]:
        if feat in df.columns:
            scatter[feat] = {
                "alive": df[df["DEATH_EVENT"] == 0][feat].tolist(),
                "dead":  df[df["DEATH_EVENT"] == 1][feat].tolist(),
            }

    # Preview: first 10 rows as list of dicts (replace NaN with None)
    preview = df.head(10).where(pd.notnull(df.head(10)), None).to_dict(orient="records")

    return {
        "filename": "uploaded_dataset.csv",
        "summary": {
            "num_records":    total,
            "num_features":   int(len(df.columns)),
            "columns":        list(df.columns),
            "death_count":    deaths,
            "survival_count": alive,
            "missing_values": int(df.isnull().sum().sum()),
            "stats":          stats,
        },
        "correlation": correlation,
        "scatter":     scatter,
        "preview":     preview,
    }


# ─────────────────────────────────────────────────────────────
# 2. MARGINAL PROBABILITY
# ─────────────────────────────────────────────────────────────

def marginal_probability(df: pd.DataFrame) -> dict:
    """
    P(Death Event = 1) — the simplest marginal probability.
    """
    total   = len(df)
    deaths  = int(df["DEATH_EVENT"].sum())
    p_death    = round(deaths / total, 4)
    p_survival = round(1.0 - p_death, 4)

    return {
        "p_death":        p_death,
        "p_survival":     p_survival,
        "deaths":         deaths,
        "survival_count": total - deaths,
        "total":          total,
    }


# ─────────────────────────────────────────────────────────────
# 3. JOINT PROBABILITY
# ─────────────────────────────────────────────────────────────

def joint_probability(df: pd.DataFrame, age_threshold: int = 60) -> dict:
    """
    P(Age > age_threshold  AND  Death Event = 1)
    """
    total   = len(df)
    joint   = int(((df["age"] > age_threshold) & (df["DEATH_EVENT"] == 1)).sum())
    p_joint         = round(joint / total, 4)
    p_age_gt        = round((df["age"] > age_threshold).sum() / total, 4)
    p_death         = round(float(df["DEATH_EVENT"].mean()), 4)

    return {
        "p_joint":            p_joint,
        "p_age_gt_threshold": p_age_gt,
        "p_death":            p_death,
        "age_threshold":      age_threshold,
        "joint_count":        joint,
        "total":              total,
    }


# ─────────────────────────────────────────────────────────────
# 4. CONDITIONAL PROBABILITY
# ─────────────────────────────────────────────────────────────

def conditional_probability(df: pd.DataFrame,
                             condition: str = "age_gt_60") -> dict:
    """
    P(Death = 1 | <condition>)

    Supported conditions:
      age_gt_60            – Age > 60
      diabetes             – Diabetes == 1
      high_blood_pressure  – High BP == 1
      anaemia              – Anaemia == 1
      smoking              – Smoking == 1
    """
    condition_map = {
        "age_gt_60":           df["age"] > 60,
        "diabetes":            df["diabetes"] == 1,
        "high_blood_pressure": df["high_blood_pressure"] == 1,
        "anaemia":             df["anaemia"] == 1,
        "smoking":             df["smoking"] == 1,
    }

    mask   = condition_map.get(condition, df["age"] > 60)
    subset = df[mask]

    p_overall_death  = round(float(df["DEATH_EVENT"].mean()), 4)
    p_cond_death     = round(float(subset["DEATH_EVENT"].mean()), 4) if len(subset) else 0.0

    # All-condition comparison for the chart
    comparison = {}
    for cname, cmask in condition_map.items():
        sub = df[cmask]
        comparison[cname] = round(float(sub["DEATH_EVENT"].mean()), 4) if len(sub) else 0.0

    return {
        "condition":          condition,
        "p_death_given_cond": p_cond_death,
        "p_overall_death":    p_overall_death,
        "condition_count":    int(mask.sum()),
        "total":              len(df),
        "comparison":         comparison,
    }


# ─────────────────────────────────────────────────────────────
# 5. MAXIMUM LIKELIHOOD ESTIMATION
# ─────────────────────────────────────────────────────────────

def mle_estimation(df: pd.DataFrame) -> dict:
    """
    MLE for a Bernoulli distribution.
    p̂ = k / n   (sample proportion of deaths)
    Also returns the log-likelihood curve for visualisation.
    """
    n      = len(df)
    k      = int(df["DEATH_EVENT"].sum())
    p_mle  = round(k / n, 6)

    # Log-likelihood: l(p) = k*log(p) + (n-k)*log(1-p)
    p_vals = np.linspace(0.01, 0.99, 200)
    log_ll = (k * np.log(p_vals) + (n - k) * np.log(1 - p_vals)).tolist()

    return {
        "p_mle":          p_mle,
        "n":              n,
        "k":              k,
        "p_values":       p_vals.tolist(),
        "log_likelihood": log_ll,
    }


# ─────────────────────────────────────────────────────────────
# 6. KL DIVERGENCE
# ─────────────────────────────────────────────────────────────

def kl_divergence(df: pd.DataFrame,
                  assumed_p_death: float = 0.30) -> dict:
    """
    KL( P_observed || P_assumed )  for two Bernoulli distributions.
    """
    p_obs  = float(df["DEATH_EVENT"].mean())
    q_asmd = assumed_p_death
    eps    = 1e-10

    kl = (p_obs * np.log((p_obs + eps) / (q_asmd + eps)) +
          (1 - p_obs) * np.log((1 - p_obs + eps) / (1 - q_asmd + eps)))

    # KL curve vs different assumed_p values
    q_range  = np.linspace(0.05, 0.95, 200)
    kl_curve = (p_obs * np.log((p_obs + eps) / (q_range + eps)) +
                (1 - p_obs) * np.log((1 - p_obs + eps) / (1 - q_range + eps)))

    return {
        "p_observed":    round(p_obs,  4),
        "p_assumed":     round(q_asmd, 4),
        "kl_divergence": round(float(kl), 6),
        "q_range":       q_range.tolist(),
        "kl_curve":      kl_curve.tolist(),
    }


# ─────────────────────────────────────────────────────────────
# 9. SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────────────────────

def generate_synthetic_data(df: pd.DataFrame,
                             n_samples: int = 20) -> dict:
    """
    Generate synthetic patient records by sampling from empirical
    feature distributions (Gaussian for continuous, categorical for binary).
    """
    np.random.seed(42)
    synthetic = {}

    for col in df.columns:
        col_data = df[col].dropna()
        if df[col].dtype in [np.float64, np.float32]:
            mu, sigma = float(col_data.mean()), float(col_data.std())
            vals = np.random.normal(mu, sigma, n_samples)
            synthetic[col] = np.round(vals, 2).tolist()
        else:
            synthetic[col] = np.random.choice(col_data.values,
                                               n_samples).tolist()

    synth_df = pd.DataFrame(synthetic)

    # Compare statistics for key features
    key_feats = ["age", "ejection_fraction", "serum_creatinine"]
    real_stats  = {}
    synth_stats = {}
    for feat in key_feats:
        if feat in df.columns:
            real_stats[feat]  = {"mean": round(float(df[feat].mean()), 4),
                                  "std":  round(float(df[feat].std()),  4)}
            synth_stats[feat] = {"mean": round(float(synth_df[feat].mean()), 4),
                                  "std":  round(float(synth_df[feat].std()),  4)}

    preview = (synth_df.head(10)
               .where(pd.notnull(synth_df.head(10)), None)
               .to_dict(orient="records"))

    return {
        "n_samples":   n_samples,
        "synthetic":   preview,
        "real_stats":  real_stats,
        "synth_stats": synth_stats,
    }
