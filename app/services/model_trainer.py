"""Real predictive-model training — the Deploy step's substance.

When a hypothesis is validated, Epiphany doesn't just *describe* a model — it
**trains a real one** on the actual dataset and measures its genuine
performance. This module:

* assembles a design matrix from the dataset's modelable columns (numeric +
  one-hot-encoded categoricals), excluding identifiers/free-text and the target;
* picks classification vs. regression from the target's type;
* trains a scikit-learn pipeline with a held-out test split and cross-validation;
* reports real metrics (ROC-AUC / accuracy / F1, or R² / RMSE) and permutation-
  free model feature importances;
* persists the fitted model (``joblib``) and its metrics (JSON) to disk so the
  artifact is a real, loadable model — not a code template.

The trained artifact's metrics flow into the findings report and the GitLab
merge request, so every deployment is backed by measured performance.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("epiphany.model")


@dataclass
class ModelArtifact:
    """The outcome of training a real model on the dataset."""

    task: str                       # "classification" | "regression"
    target: str
    algorithm: str
    n_rows: int
    n_features: int
    feature_names: list[str]
    metrics: dict[str, float]
    feature_importances: list[dict[str, Any]]
    model_path: str | None = None
    metrics_path: str | None = None
    notes: list[str] = field(default_factory=list)

    def headline_metric(self) -> tuple[str, float]:
        """Return the single most representative metric (name, value)."""
        for key in ("roc_auc", "r2", "accuracy", "f1"):
            if key in self.metrics:
                return key, self.metrics[key]
        if self.metrics:
            k = next(iter(self.metrics))
            return k, self.metrics[k]
        return ("n/a", float("nan"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "target": self.target,
            "algorithm": self.algorithm,
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "feature_importances": self.feature_importances,
            "model_path": self.model_path,
            "metrics_path": self.metrics_path,
            "notes": self.notes,
        }


# Columns we never feed a model: identifiers and free text carry no signal and
# would leak/overfit. Datetime is dropped here (could be engineered later).
_EXCLUDED_ROLES = {"identifier", "text", "datetime"}


def _select_columns(
    df: pd.DataFrame, profile: Any, target: str
) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) usable as model features."""
    numeric, categorical = [], []
    for col in profile.columns:
        if col.name == target or col.role in _EXCLUDED_ROLES:
            continue
        if col.name not in df.columns:
            continue
        if col.role == "numeric":
            numeric.append(col.name)
        elif col.role in ("binary", "categorical"):
            # A numeric-dtyped binary stays numeric; an object one is encoded.
            if pd.api.types.is_numeric_dtype(df[col.name]):
                numeric.append(col.name)
            else:
                categorical.append(col.name)
    return numeric, categorical


def train_model(
    df: pd.DataFrame,
    profile: Any,
    target: str,
    artifacts_dir: str = "artifacts",
    *,
    random_state: int = 42,
) -> ModelArtifact:
    """Train and evaluate a real predictive model for ``target`` on ``df``.

    Args:
        df: The real dataset.
        profile: A :class:`app.services.data_port.DataProfile`.
        target: The column to predict.
        artifacts_dir: Directory to write the fitted model + metrics into.

    Returns:
        A :class:`ModelArtifact` with measured metrics and saved file paths.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        GradientBoostingRegressor,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    target_col = profile.column(target)
    if target_col is None or target not in df.columns:
        raise ValueError(f"target column '{target}' not found in the dataset")

    numeric_cols, categorical_cols = _select_columns(df, profile, target)
    if not numeric_cols and not categorical_cols:
        raise ValueError("no usable feature columns to train a model")

    work = df[numeric_cols + categorical_cols + [target]].dropna(subset=[target]).copy()
    y_raw = work[target]
    notes: list[str] = []

    # Decide the task from the target's role.
    is_classification = target_col.role in ("binary", "categorical")
    if is_classification:
        task = "classification"
        # Encode labels to integers deterministically.
        classes = sorted(y_raw.dropna().unique().tolist(), key=str)
        class_map = {c: i for i, c in enumerate(classes)}
        y = y_raw.map(class_map).astype(int)
        notes.append(f"Classes: {', '.join(str(c) for c in classes)}")
    else:
        task = "regression"
        y = pd.to_numeric(y_raw, errors="coerce")
        work = work.loc[y.notna()]
        y = y.loc[work.index]

    X = work[numeric_cols + categorical_cols]

    # Preprocessing: impute + scale numerics, impute + one-hot categoricals.
    transformers = []
    if numeric_cols:
        transformers.append(
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric_cols)
        )
    if categorical_cols:
        transformers.append(
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=20)),
            ]), categorical_cols)
        )
    preprocessor = ColumnTransformer(transformers, remainder="drop")

    if task == "classification":
        estimator = GradientBoostingClassifier(random_state=random_state)
        scoring = "roc_auc" if len(class_map) == 2 else "accuracy"
    else:
        estimator = GradientBoostingRegressor(random_state=random_state)
        scoring = "r2"

    pipeline = Pipeline([("prep", preprocessor), ("model", estimator)])

    stratify = y if (task == "classification" and y.nunique() > 1) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=stratify
    )
    pipeline.fit(X_train, y_train)

    metrics = _evaluate(
        pipeline, X_train, X_test, y_train, y_test, task,
        n_classes=(y.nunique() if task == "classification" else 0),
    )

    # Cross-validated score for a robustness check (best-effort).
    try:
        cv = cross_val_score(pipeline, X, y, cv=min(5, max(2, y.nunique() if task == "classification" else 5)), scoring=scoring)
        metrics[f"cv_{scoring}_mean"] = round(float(cv.mean()), 4)
        metrics[f"cv_{scoring}_std"] = round(float(cv.std()), 4)
    except Exception as exc:  # noqa: BLE001 - CV is a nice-to-have
        notes.append(f"cross-validation skipped: {exc}")

    feature_names = _expanded_feature_names(pipeline, numeric_cols, categorical_cols)
    importances = _feature_importances(pipeline, feature_names)

    artifact = ModelArtifact(
        task=task,
        target=target,
        algorithm=type(estimator).__name__,
        n_rows=int(len(work)),
        n_features=len(feature_names),
        feature_names=feature_names,
        metrics=metrics,
        feature_importances=importances,
        notes=notes,
    )
    _persist(artifact, pipeline, artifacts_dir, target)
    return artifact


def _evaluate(
    pipeline, X_train, X_test, y_train, y_test, task: str, n_classes: int
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )

    metrics: dict[str, float] = {}
    if task == "classification":
        preds = pipeline.predict(X_test)
        metrics["accuracy"] = round(float(accuracy_score(y_test, preds)), 4)
        metrics["f1"] = round(
            float(f1_score(y_test, preds, average="binary" if n_classes == 2 else "macro")),
            4,
        )
        if n_classes == 2 and hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(X_test)[:, 1]
            try:
                metrics["roc_auc"] = round(float(roc_auc_score(y_test, proba)), 4)
            except ValueError:
                pass
    else:
        preds = pipeline.predict(X_test)
        metrics["r2"] = round(float(r2_score(y_test, preds)), 4)
        metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, preds))), 4)
        metrics["mae"] = round(float(mean_absolute_error(y_test, preds)), 4)
    metrics["test_rows"] = int(len(X_test))
    metrics["train_rows"] = int(len(X_train))
    return metrics


def _expanded_feature_names(
    pipeline, numeric_cols: list[str], categorical_cols: list[str]
) -> list[str]:
    """Recover post-encoding feature names from the fitted ColumnTransformer."""
    try:
        prep = pipeline.named_steps["prep"]
        return list(prep.get_feature_names_out())
    except Exception:  # noqa: BLE001 - fall back to raw column names
        return numeric_cols + categorical_cols


def _feature_importances(pipeline, feature_names: list[str]) -> list[dict[str, Any]]:
    model = pipeline.named_steps["model"]
    importances = getattr(model, "feature_importances_", None)
    if importances is None or len(importances) != len(feature_names):
        return []
    ranked = sorted(
        ({"feature": _clean_name(n), "importance": round(float(v), 4)}
         for n, v in zip(feature_names, importances)),
        key=lambda d: d["importance"],
        reverse=True,
    )
    return ranked[:15]


def _clean_name(name: str) -> str:
    """Strip the ColumnTransformer's ``num__`` / ``cat__`` prefixes."""
    for prefix in ("num__", "cat__"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _persist(
    artifact: ModelArtifact, pipeline, artifacts_dir: str, target: str
) -> None:
    """Save the fitted pipeline (.pkl) and metrics (.json) to disk."""
    import re

    import joblib

    out = Path(artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"\W+", "_", str(target)).strip("_").lower() or "target"
    model_path = out / f"model_{slug}.pkl"
    metrics_path = out / f"model_{slug}_metrics.json"
    try:
        joblib.dump(pipeline, model_path)
        artifact.model_path = str(model_path)
    except Exception as exc:  # noqa: BLE001 - persistence is best-effort
        logger.warning("Could not persist model: %s", exc)
        artifact.notes.append(f"model not saved: {exc}")
    try:
        metrics_path.write_text(json.dumps(artifact.as_dict(), indent=2, default=str))
        artifact.metrics_path = str(metrics_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not persist metrics: %s", exc)
