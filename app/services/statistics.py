"""Adaptive statistical testing — the real Validate step.

Given a feature column and a target column, this module *chooses the right test*
from the data's actual shape and runs it with SciPy:

* categorical feature × categorical/binary target  → **Chi-Square** test
* numeric feature × 2-class target                 → **Welch's t-test**
* numeric feature × 3+-class target                → **one-way ANOVA**
* categorical feature × numeric target             → **ANOVA / t-test**
* numeric feature × numeric target                 → **Pearson correlation**

Every statistic and p-value is computed from the real values handed in — there
is no synthesis and no rigged effect. The result also carries an effect size
(Cohen's d, Cramér's V, η², or |r|) so a *statistically* significant finding can
be judged for *practical* significance too.

The functions here are deliberately dependency-light (NumPy + SciPy only) and
pure, so the hardened sandbox worker can call them in an isolated process.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import stats

# Test identifiers.
CHI_SQUARE = "chi_square"
T_TEST = "t_test"
ANOVA = "anova"
CORRELATION = "correlation"

VALID_TESTS = frozenset({CHI_SQUARE, T_TEST, ANOVA, CORRELATION})


def choose_test(
    feature_role: str,
    target_role: str,
    n_target_classes: int,
    n_feature_classes: int,
) -> str:
    """Pick the appropriate test from the feature/target roles.

    Roles are the strings from :mod:`app.services.data_port`
    (``numeric`` | ``binary`` | ``categorical`` | ...).
    """
    feature_numeric = feature_role == "numeric"
    target_numeric = target_role == "numeric"

    if not target_numeric:
        # Classification-style target (binary / categorical).
        if feature_numeric:
            return T_TEST if n_target_classes <= 2 else ANOVA
        return CHI_SQUARE

    # Continuous numeric target.
    if feature_numeric:
        return CORRELATION
    # Categorical feature vs numeric target → compare group means.
    return T_TEST if n_feature_classes <= 2 else ANOVA


def run_statistical_test(
    test: str,
    feature_values: Sequence[Any],
    target_values: Sequence[Any],
    alpha: float = 0.05,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Run ``test`` on real, row-aligned ``feature``/``target`` values.

    Returns a JSON-serialisable result dict with the statistic, p-value,
    significance verdict, sample size, an effect size, and a human-readable
    summary. Raises :class:`ValueError` on unrunnable inputs.
    """
    if test not in VALID_TESTS:
        raise ValueError(f"unknown test '{test}'")

    feat = list(feature_values)
    targ = list(target_values)
    if len(feat) != len(targ):
        raise ValueError("feature and target must be the same length")
    if len(feat) < 3:
        raise ValueError("not enough rows to run a statistical test")

    if test == CHI_SQUARE:
        result = _chi_square(feat, targ, threshold)
    elif test == T_TEST:
        result = _t_test(feat, targ)
    elif test == ANOVA:
        result = _anova(feat, targ)
    else:  # CORRELATION
        result = _correlation(feat, targ)

    result["test"] = test
    result["alpha"] = alpha
    result["is_significant"] = bool(
        result.get("p_value") is not None and result["p_value"] < alpha
    )
    result["sample_size"] = len(feat)
    return result


# ── Numeric/grouping orientation helpers ────────────────────────────────
def _is_numeric_array(values: list[Any]) -> bool:
    try:
        arr = np.asarray(values, dtype=float)
    except (ValueError, TypeError):
        return False
    if np.isnan(arr).all():
        return False
    # Treat a 2-valued numeric column as categorical for grouping purposes.
    return len(np.unique(arr[~np.isnan(arr)])) > 2


def _orient(feat: list[Any], targ: list[Any]) -> tuple[np.ndarray, list[Any]]:
    """Return (numeric_array, group_labels) regardless of which side is numeric."""
    if _is_numeric_array(feat):
        return np.asarray(feat, dtype=float), [str(t) for t in targ]
    if _is_numeric_array(targ):
        return np.asarray(targ, dtype=float), [str(f) for f in feat]
    raise ValueError("a numeric column is required for this test")


# ── Individual tests ────────────────────────────────────────────────────
def _chi_square(
    feat: list[Any], targ: list[Any], threshold: float | None
) -> dict[str, Any]:
    """Chi-Square test of independence on a real contingency table.

    If the feature is numeric, it is split at ``threshold`` (or its median) into
    a high/low category so a contingency table can be built.
    """
    feat_arr = feat
    if _is_numeric_array(feat):
        numeric = np.asarray(feat, dtype=float)
        cut = float(threshold) if threshold is not None else float(np.nanmedian(numeric))
        feat_arr = np.where(numeric > cut, f">{cut:g}", f"<={cut:g}")
        feat_labels = [str(v) for v in feat_arr]
    else:
        feat_labels = [str(v) for v in feat]

    targ_labels = [str(v) for v in targ]
    # Build the contingency table.
    rows = sorted(set(feat_labels))
    cols = sorted(set(targ_labels))
    if len(rows) < 2 or len(cols) < 2:
        raise ValueError("chi-square needs at least 2 categories on each axis")
    row_idx = {r: i for i, r in enumerate(rows)}
    col_idx = {c: i for i, c in enumerate(cols)}
    table = np.zeros((len(rows), len(cols)), dtype=int)
    for f, t in zip(feat_labels, targ_labels):
        table[row_idx[f], col_idx[t]] += 1

    chi2, p_value, dof, _ = stats.chi2_contingency(table)
    n = int(table.sum())
    # Cramér's V effect size.
    min_dim = min(table.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else 0.0
    return {
        "statistic": float(chi2),
        "statistic_name": "chi2",
        "p_value": float(p_value),
        "dof": int(dof),
        "effect_size": round(cramers_v, 4),
        "effect_name": "cramers_v",
        "contingency": table.tolist(),
        "row_labels": rows,
        "col_labels": cols,
        "summary": (
            f"χ²={chi2:.2f}, p={p_value:.4g} across a {len(rows)}×{len(cols)} "
            f"table (Cramér's V={cramers_v:.3f})."
        ),
    }


def _t_test(feat: list[Any], targ: list[Any]) -> dict[str, Any]:
    """Welch's t-test comparing a numeric variable across two groups."""
    numeric, groups = _orient(feat, targ)
    mask = ~np.isnan(numeric)
    numeric = numeric[mask]
    groups = [g for g, keep in zip(groups, mask) if keep]

    labels = sorted(set(groups))
    if len(labels) != 2:
        raise ValueError(f"t-test needs exactly 2 groups, found {len(labels)}")
    a = numeric[np.array(groups) == labels[0]]
    b = numeric[np.array(groups) == labels[1]]
    if len(a) < 2 or len(b) < 2:
        raise ValueError("each group needs at least 2 observations")

    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
    # Cohen's d (pooled).
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) or 1.0
    cohens_d = float((a.mean() - b.mean()) / pooled_sd)
    return {
        "statistic": float(t_stat),
        "statistic_name": "t",
        "p_value": float(p_value),
        "effect_size": round(abs(cohens_d), 4),
        "effect_name": "cohens_d",
        "group_means": {labels[0]: float(a.mean()), labels[1]: float(b.mean())},
        "group_sizes": {labels[0]: int(len(a)), labels[1]: int(len(b))},
        "summary": (
            f"t={t_stat:.2f}, p={p_value:.4g}. Means: {labels[0]}={a.mean():.2f} "
            f"vs {labels[1]}={b.mean():.2f} (Cohen's d={abs(cohens_d):.2f})."
        ),
    }


def _anova(feat: list[Any], targ: list[Any]) -> dict[str, Any]:
    """One-way ANOVA comparing a numeric variable across 3+ groups."""
    numeric, groups = _orient(feat, targ)
    mask = ~np.isnan(numeric)
    numeric = numeric[mask]
    groups = np.array([g for g, keep in zip(groups, mask) if keep])

    labels = sorted(set(groups.tolist()))
    samples = [numeric[groups == lab] for lab in labels]
    samples = [s for s in samples if len(s) >= 2]
    if len(samples) < 2:
        raise ValueError("ANOVA needs at least 2 groups with >=2 observations")

    f_stat, p_value = stats.f_oneway(*samples)
    # η² effect size = SS_between / SS_total.
    grand_mean = numeric.mean()
    ss_total = float(((numeric - grand_mean) ** 2).sum()) or 1.0
    ss_between = float(sum(len(s) * (s.mean() - grand_mean) ** 2 for s in samples))
    eta_sq = ss_between / ss_total
    return {
        "statistic": float(f_stat),
        "statistic_name": "F",
        "p_value": float(p_value),
        "effect_size": round(eta_sq, 4),
        "effect_name": "eta_squared",
        "group_means": {
            str(lab): float(s.mean()) for lab, s in zip(labels, samples)
        },
        "summary": (
            f"F={f_stat:.2f}, p={p_value:.4g} across {len(samples)} groups "
            f"(η²={eta_sq:.3f})."
        ),
    }


def _correlation(feat: list[Any], targ: list[Any]) -> dict[str, Any]:
    """Pearson correlation between two numeric variables (+ Spearman)."""
    x = np.asarray(feat, dtype=float)
    y = np.asarray(targ, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3:
        raise ValueError("correlation needs at least 3 paired observations")

    r, p_value = stats.pearsonr(x, y)
    rho, _ = stats.spearmanr(x, y)
    direction = "positive" if r >= 0 else "negative"
    return {
        "statistic": float(r),
        "statistic_name": "pearson_r",
        "p_value": float(p_value),
        "effect_size": round(abs(float(r)), 4),
        "effect_name": "abs_r",
        "spearman_rho": float(rho),
        "summary": (
            f"Pearson r={r:.3f} ({direction}), p={p_value:.4g} "
            f"(Spearman ρ={rho:.3f}, n={len(x)})."
        ),
    }
