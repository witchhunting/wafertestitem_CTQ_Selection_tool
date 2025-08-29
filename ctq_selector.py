import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import time
import warnings
from typing import List, Optional

warnings.filterwarnings('ignore')


def pointbiserial_bootstrap(
    df: pd.DataFrame,
    pos_label: str,
    y_col: str,
    test_item_columns: List[str],
    seed: int = 42,
    n_bootstrap: int = 1000,
    k_neg: int = 1,
    n_max_neg: int = 200_000,
    save_prefix: Optional[str] = None,
    tqdm_class=tqdm,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run high-speed point-biserial correlation analysis.

    Parameters
    ----------
    df : DataFrame
        Input data containing the label column and feature columns.
    pos_label : str
        Defect type to analyze. Rows with ``pos_label`` are treated as
        positive samples, "PASS" rows as negative samples, and others are
        ignored.
    y_col : str
        Name of the label column in ``df``.
    test_item_columns : List[str]
        Feature columns to evaluate.
    seed : int, optional
        Random seed for reproducibility, by default 42.
    n_bootstrap : int, optional
        Number of bootstrap iterations, by default 1000.
    k_neg : int, optional
        Negative sampling ratio relative to positives, by default 1.
    n_max_neg : int, optional
        Maximum number of negative samples to draw, by default 200_000.
    save_prefix : str, optional
        If provided, CSV files with results will be written using this prefix.
    tqdm_class : tqdm class, optional
        Progress bar implementation to use, by default ``tqdm``.

    Returns
    -------
    DataFrame
        Statistics for each valid feature.
    """

    start_time = time.time()
    if verbose:
        print("=== High-speed Point-biserial Correlation Analysis ===")
        print(f"Target defect: {pos_label}")

    lab = df[y_col].astype("string").str.strip().str.upper()
    y_all = np.where(
        lab == pos_label.upper(),
        1,
        np.where(lab == "PASS", 0, -1)
    ).astype(np.int8)

    pos_idx_all = np.where(y_all == 1)[0]
    pass_idx_all = np.where(y_all == 0)[0]
    n_pos = len(pos_idx_all)
    n_pass = len(pass_idx_all)
    n_other = np.sum(y_all == -1)

    if verbose:
        print("Class distribution:")
        print(f"  - {pos_label}: {n_pos:,}")
        print(f"  - PASS: {n_pass:,}")
        print(f"  - Other: {n_other:,} (ignored)")

    valid_idx = np.concatenate([pos_idx_all, pass_idx_all])
    df_valid = df.iloc[valid_idx]

    X = df_valid[test_item_columns].to_numpy(dtype=np.float64)
    y_valid = y_all[valid_idx]
    X[~np.isfinite(X)] = np.nan

    valid_columns = []
    col_masks = []
    for j, col in enumerate(test_item_columns):
        col_data = X[:, j]
        mask = ~np.isnan(col_data)
        if np.sum(mask) >= 10 and np.unique(col_data[mask]).size > 1:
            valid_columns.append(col)
            col_masks.append(mask)
        else:
            col_masks.append(None)

    if verbose:
        print(f"Valid features: {len(valid_columns)}/{len(test_item_columns)}")
    preprocessing_time = time.time() - start_time
    if verbose:
        print(f"Preprocessing completed ({preprocessing_time:.2f}s)")

    def fast_bootstrap(seed_val: int) -> np.ndarray:
        rng = np.random.default_rng(seed_val)
        n_pos = np.sum(y_valid == 1)
        if n_pos == 0:
            return np.zeros(len(valid_columns))

        n_neg_target = min(n_pos * k_neg, np.sum(y_valid == 0), n_max_neg)

        pos_idx = np.where(y_valid == 1)[0]
        neg_idx = rng.choice(np.where(y_valid == 0)[0], size=n_neg_target, replace=False)

        sel_idx = np.concatenate([pos_idx, neg_idx])
        rng.shuffle(sel_idx)

        y_sample = y_valid[sel_idx]
        X_sample = X[sel_idx, :]

        corrs = np.zeros(len(valid_columns))
        for j, col in enumerate(valid_columns):
            mask = col_masks[j]
            if mask is None:
                continue
            col_data = X_sample[:, j]
            finite_mask = ~np.isnan(col_data)
            if np.sum(finite_mask) < 10:
                continue
            x_clean = col_data[finite_mask]
            y_clean = y_sample[finite_mask]
            if np.unique(x_clean).size <= 1:
                continue
            corr = np.corrcoef(x_clean, y_clean)[0, 1]
            corrs[j] = corr if np.isfinite(corr) else 0.0
        return corrs

    bootstrap_start = time.time()
    if verbose:
        print(f"=== Bootstrap (n={n_bootstrap}) ===")
    seeds = [seed + i for i in range(n_bootstrap)]
    correlations_matrix = np.array(
        process_map(
            fast_bootstrap,
            seeds,
            max_workers=None,
            desc="Fast Bootstrap",
            unit="iter",
            tqdm_class=tqdm_class,
            chunksize=max(10, n_bootstrap // 100),
        )
    )
    bootstrap_time = time.time() - bootstrap_start
    if verbose:
        print(f"Bootstrap done: {correlations_matrix.shape} ({bootstrap_time:.2f}s)")

    analysis_start = time.time()
    if verbose:
        print("=== Statistics ===")
    pb_mean = np.nanmean(correlations_matrix, axis=0)
    pb_median = np.nanmedian(correlations_matrix, axis=0)
    pb_std = np.nanstd(correlations_matrix, axis=0)
    pb_q25 = np.nanpercentile(correlations_matrix, 25, axis=0)
    pb_q75 = np.nanpercentile(correlations_matrix, 75, axis=0)
    ci_lower = np.nanpercentile(correlations_matrix, 2.5, axis=0)
    ci_upper = np.nanpercentile(correlations_matrix, 97.5, axis=0)

    stability = []
    for i in range(correlations_matrix.shape[1]):
        feature_corrs = correlations_matrix[:, i]
        feature_corrs = feature_corrs[np.isfinite(feature_corrs)]
        if len(feature_corrs) == 0:
            stability.append(0.0)
        else:
            median_sign = np.sign(np.median(feature_corrs))
            if median_sign == 0:
                stability.append(0.0)
            else:
                consistency = np.mean(np.sign(feature_corrs) == median_sign)
                stability.append(consistency)
    stability = np.array(stability)

    effect_size = np.abs(pb_median)
    significant = (ci_lower > 0) | (ci_upper < 0)
    analysis_time = time.time() - analysis_start

    results_df = pd.DataFrame(
        {
            "feature": valid_columns,
            "pb_mean": pb_mean,
            "pb_median": pb_median,
            "pb_std": pb_std,
            "pb_q25": pb_q25,
            "pb_q75": pb_q75,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "effect_size": effect_size,
            "stability": stability,
            "significant": significant,
            "direction": np.where(pb_median > 0, "positive", "negative"),
        }
    ).sort_values("effect_size", ascending=False).reset_index(drop=True)

    if verbose:
        print(f"Point-biserial range: [{np.nanmin(pb_median):.4f}, {np.nanmax(pb_median):.4f}]")
        print(f"Significant features: {np.sum(significant)}")
        print(f"Mean stability: {np.nanmean(stability):.3f}")

    if save_prefix:
        results_df.to_csv(f"{save_prefix}_pointbiserial_full.csv", index=False)
        results_df.head(50).to_csv(f"{save_prefix}_pointbiserial_top50.csv", index=False)
        sig_features = results_df[results_df["significant"]].copy()
        if len(sig_features) > 0:
            sig_features.to_csv(
                f"{save_prefix}_pointbiserial_significant.csv", index=False
            )

    total_time = time.time() - start_time
    if verbose:
        print(f"=== Analysis done ({total_time:.2f}s) ===")
    return results_df
