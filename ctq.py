"""CTQ selection workflow in a single script.

This file demonstrates how to compute point-biserial correlations for many
test parameters and pick those that are strongly related to a failure label.

The implementation avoids command line options; instead tweak the variables
at the bottom of the file and run ``python ctq.py``.

The heavy step of running a bootstrap correlation for hundreds or thousands of
parameters is parallelised with :func:`tqdm.contrib.concurrent.process_map`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map


def generate_data(n_samples: int = 100, n_features: int = 10, random_state: int | None = 0) -> pd.DataFrame:
    """Create a synthetic dataset.

    Parameters
    ----------
    n_samples:
        Number of rows in the dataset.
    n_features:
        Number of numeric parameter columns to create.  Set this to ``2000`` in
        a real scenario.
    random_state:
        Seed for reproducibility.
    """

    rng = np.random.default_rng(random_state)
    data = {"label": rng.choice(["M0C-GT Short", "other"], size=n_samples)}
    for i in range(n_features):
        # Generate random values for each parameter column
        data[f"para_{i}"] = rng.normal(size=n_samples)
    return pd.DataFrame(data)


def preprocess(df: pd.DataFrame, pos_label: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Convert labels to 0/1 and separate parameter columns.

    Returns
    -------
    x:
        Array of 0/1 values where 1 represents ``pos_label``.
    features:
        DataFrame containing only the parameter columns.
    """

    x = (df["label"] == pos_label).astype(int).to_numpy()
    features = df.drop(columns=["label"])
    return x, features


def pointbiserial_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Vectorised point-biserial correlation."""

    mask = x.astype(bool)
    y1 = y[mask]
    y0 = y[~mask]
    n1 = y1.size
    n0 = y0.size
    if n1 == 0 or n0 == 0:
        # If one of the classes is missing, correlation is undefined.
        return 0.0
    mean1 = y1.mean()
    mean0 = y0.mean()
    std = y.std(ddof=1)
    return (mean1 - mean0) / std * np.sqrt((n1 * n0) / ((n1 + n0) ** 2))


def pointbiserial_bootstrap(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, rng: int | None = None) -> tuple[float, float, float]:
    """Estimate correlation with a 95% bootstrap confidence interval.

    ``ci_low`` and ``ci_high`` are the lower and upper bounds of the interval.
    If they do not contain zero, the correlation is statistically significant.
    """

    rng = np.random.default_rng(rng)
    n = len(x)
    stats = np.empty(n_boot, float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[i] = pointbiserial_corr(x[idx], y[idx])
    corr = pointbiserial_corr(x, y)
    ci_low, ci_high = np.percentile(stats, [2.5, 97.5])
    return corr, ci_low, ci_high


def _bootstrap_column(args: tuple[str, np.ndarray, np.ndarray, int]) -> tuple[str, float, float, float]:
    """Helper for parallel processing of a single column."""

    name, x, y, n_boot = args
    corr, ci_low, ci_high = pointbiserial_bootstrap(x, y, n_boot)
    return name, corr, ci_low, ci_high


def correlate_features(x: np.ndarray, features: pd.DataFrame, n_boot: int, threshold: float) -> list[tuple[str, float, float, float]]:
    """Run bootstrap correlation on each feature in parallel and filter results."""

    # Prepare (name, x, column_values, n_boot) tuples for each parameter.
    jobs = [(name, x, features[name].to_numpy(), n_boot) for name in features.columns]

    # ``process_map`` distributes the work across processes for speed.
    results = process_map(_bootstrap_column, jobs, chunksize=10)

    # Keep only items whose correlation exceeds the threshold.
    selected = [res for res in results if res[1] >= threshold]
    return selected


# ---- Example workflow ----
# Configure options here instead of passing command line arguments.
pos_label = "M0C-GT Short"
bootstrap_iterations = 200  # Increase for more stable intervals
corr_threshold = 0.1

# Step 1: generate data (set n_features=2000 in a real dataset)
frame = generate_data(n_samples=200, n_features=20, random_state=0)

# Step 2: preprocessing
binary_labels, feature_frame = preprocess(frame, pos_label)

# Step 3: correlation with bootstrap for every parameter in parallel
strong_items = correlate_features(binary_labels, feature_frame, bootstrap_iterations, corr_threshold)

# Display the parameters that show notable correlation with the failure label
print("Items with correlation >=", corr_threshold)
for name, corr, ci_low, ci_high in strong_items:
    print(f"{name}: corr={corr:.3f}, 95% CI=({ci_low:.3f}, {ci_high:.3f})")

