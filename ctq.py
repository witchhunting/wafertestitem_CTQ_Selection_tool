import numpy as np
import pandas as pd


def generate_data(n=20, random_state=0):
    """Create a small synthetic dataset."""
    rng = np.random.default_rng(random_state)
    df = pd.DataFrame(
        {
            "label": rng.choice(["M0C-GT Short", "other"], size=n),
            "value": rng.normal(size=n),
        }
    )
    return df


def preprocess(df, pos_label):
    """Convert labels to 0/1 based on ``pos_label``."""
    x = (df["label"] == pos_label).astype(int).to_numpy()
    y = df["value"].to_numpy()
    return x, y


def pointbiserial_corr(x, y):
    """Vectorized point-biserial correlation."""
    mask = x.astype(bool)
    y1 = y[mask]
    y0 = y[~mask]
    n1 = y1.size
    n0 = y0.size
    if n1 == 0 or n0 == 0:
        return 0.0
    mean1 = y1.mean()
    mean0 = y0.mean()
    std = y.std(ddof=1)
    return (mean1 - mean0) / std * np.sqrt((n1 * n0) / ((n1 + n0) ** 2))


def pointbiserial_bootstrap(x, y, n_boot=1000, rng=None):
    """Bootstrap confidence interval for the correlation."""
    rng = np.random.default_rng(rng)
    n = len(x)
    stats = np.empty(n_boot, float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[i] = pointbiserial_corr(x[idx], y[idx])
    corr = pointbiserial_corr(x, y)
    ci_low, ci_high = np.percentile(stats, [2.5, 97.5])
    return corr, ci_low, ci_high


# ---- Example workflow ----
# Set options directly in the code
pos_label = "M0C-GT Short"
bootstrap_iterations = 1000

# Step 1: generate data
frame = generate_data()

# Step 2: preprocessing
binary, values = preprocess(frame, pos_label)

# Step 3: correlation with bootstrap
corr, ci_low, ci_high = pointbiserial_bootstrap(
    binary, values, bootstrap_iterations, rng=0
)

print("Correlation:", float(corr))
print("95% CI:", (float(ci_low), float(ci_high)))
