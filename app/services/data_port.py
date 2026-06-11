"""Real data access + profiling for Epiphany.

This is the foundation that makes Epiphany *real*: every step of the agent loop
reasons over a genuine :class:`pandas.DataFrame`, never a fabricated one. The
frame is sourced transparently from whichever backend is configured:

* **Live** — when Elastic is configured, rows are pulled from the telemetry
  index into a DataFrame.
* **Local** — otherwise, a real dataset is read from ``DATA_CSV_PATH`` (a CSV or
  Parquet file). This is the "anyone can use it" path: drop in any tabular file
  and the agent does real data science on it.

On top of the frame, :class:`DataPort` profiles every column (inferring whether
it is numeric, binary, categorical, datetime, an identifier, or free text),
proposes candidate targets/features, runs real group aggregations, and ranks
features by their genuine univariate association with a target. None of these
numbers are synthesised — they are computed from the data in front of us.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import Settings

logger = logging.getLogger("epiphany.data")

# Roles a column can play in the analysis.
ROLE_NUMERIC = "numeric"        # continuous / discrete number usable as a feature
ROLE_BINARY = "binary"          # exactly two outcomes — an ideal classification target
ROLE_CATEGORICAL = "categorical"  # low-cardinality labels
ROLE_DATETIME = "datetime"
ROLE_IDENTIFIER = "identifier"  # ids / emails / names — excluded from analysis
ROLE_TEXT = "text"              # high-cardinality free text — excluded from analysis

# A categorical column with more distinct values than this is treated as an
# identifier/text column and excluded from modelling.
_MAX_CATEGORY_CARDINALITY = 20


@dataclass
class ColumnProfile:
    """Everything the agent needs to know about one column."""

    name: str
    dtype: str
    role: str
    cardinality: int
    missing: int
    sample_values: list[Any] = field(default_factory=list)
    # Numeric summary (None for non-numeric columns).
    mean: float | None = None
    std: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    median: float | None = None

    @property
    def is_modelable_feature(self) -> bool:
        return self.role in (ROLE_NUMERIC, ROLE_BINARY, ROLE_CATEGORICAL)

    @property
    def is_candidate_target(self) -> bool:
        # Binary columns make the cleanest targets; low-card categoricals and
        # plain numerics are also valid (classification / regression).
        return self.role in (ROLE_BINARY, ROLE_CATEGORICAL, ROLE_NUMERIC)

    def as_dict(self) -> dict[str, Any]:
        d = {
            "name": self.name,
            "type": self.dtype,
            "role": self.role,
            "cardinality": self.cardinality,
            "missing": self.missing,
            "sample_values": self.sample_values,
        }
        if self.role == ROLE_NUMERIC:
            d.update(
                mean=self.mean, std=self.std, min=self.minimum,
                max=self.maximum, median=self.median,
            )
        return d


def infer_role(series: pd.Series, name: str) -> str:
    """Infer a column's analytical role from its values (not its name alone)."""
    non_null = series.dropna()
    n = len(non_null)
    if n == 0:
        return ROLE_TEXT
    nunique = int(non_null.nunique())

    if pd.api.types.is_datetime64_any_dtype(series):
        return ROLE_DATETIME

    if pd.api.types.is_numeric_dtype(series):
        # Numeric with two distinct values is a binary target/feature.
        if nunique <= 2:
            return ROLE_BINARY
        return ROLE_NUMERIC

    # Object / string / categorical columns.
    lowered = str(name).lower()
    looks_like_id = (
        lowered.endswith("_id")
        or lowered == "id"
        or lowered in {"email", "name", "uuid", "guid"}
        or "email" in lowered
    )
    unique_ratio = nunique / n
    if nunique == 2:
        return ROLE_BINARY
    if looks_like_id or unique_ratio > 0.5:
        return ROLE_IDENTIFIER
    if nunique <= _MAX_CATEGORY_CARDINALITY:
        return ROLE_CATEGORICAL
    return ROLE_TEXT


def profile_column(series: pd.Series, name: str) -> ColumnProfile:
    """Build a :class:`ColumnProfile` for a single column."""
    role = infer_role(series, name)
    non_null = series.dropna()
    prof = ColumnProfile(
        name=name,
        dtype=str(series.dtype),
        role=role,
        cardinality=int(non_null.nunique()),
        missing=int(series.isna().sum()),
        sample_values=[_jsonable(v) for v in non_null.unique()[:5]],
    )
    if role == ROLE_NUMERIC and len(non_null):
        prof.mean = float(non_null.mean())
        prof.std = float(non_null.std())
        prof.minimum = float(non_null.min())
        prof.maximum = float(non_null.max())
        prof.median = float(non_null.median())
    return prof


def _jsonable(value: Any) -> Any:
    """Coerce a numpy/pandas scalar to a plain JSON-serialisable Python value."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


@dataclass
class DataProfile:
    """A complete, real description of the dataset under analysis."""

    source: str            # index name or file path
    mode: str              # "live" (Elastic) or "local" (file)
    row_count: int
    columns: list[ColumnProfile]

    def field_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def by_role(self, *roles: str) -> list[ColumnProfile]:
        return [c for c in self.columns if c.role in roles]

    def candidate_targets(self) -> list[ColumnProfile]:
        """Rank likely prediction targets with a domain-agnostic heuristic.

        Combines three signals that hold across domains, so the agent picks a
        sensible target whether the data is churn, housing, or sensor logs:

        * **Name hints** — columns literally named like a target (target, label,
          class, outcome, y, ...).
        * **Position** — the last column is the target by overwhelming
          convention in tabular ML datasets.
        * **Role** — binary/categorical outcomes are cleaner targets than a
          continuous numeric, but this never overrides a strong name/position
          signal (so a numeric ``price`` or ``disease_progression`` still wins).
        """
        targets = [c for c in self.columns if c.is_candidate_target]
        if not targets:
            return []
        last_index = len(self.columns) - 1
        position = {c.name: i for i, c in enumerate(self.columns)}
        role_score = {ROLE_BINARY: 20, ROLE_CATEGORICAL: 12, ROLE_NUMERIC: 6}
        name_hints = {
            "target", "label", "class", "outcome", "y", "result", "response",
            "prediction", "status", "category",
        }

        def score(c: ColumnProfile) -> int:
            s = role_score.get(c.role, 0)
            if position[c.name] == last_index:
                s += 50  # last-column-is-target convention
            tokens = set(re.split(r"[\W_]+", c.name.lower()))
            if tokens & name_hints:
                s += 60
            return s

        return sorted(targets, key=score, reverse=True)

    def candidate_features(self, target: str | None = None) -> list[ColumnProfile]:
        return [
            c for c in self.columns
            if c.is_modelable_feature and c.name != target
        ]

    def column(self, name: str) -> ColumnProfile | None:
        for c in self.columns:
            if c.name == name:
                return c
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "mode": self.mode,
            "row_count": self.row_count,
            "fields": [c.as_dict() for c in self.columns],
            "candidate_targets": [c.name for c in self.candidate_targets()],
        }


class DataPort:
    """Loads and profiles the real dataset the agent reasons over.

    The frame is loaded once and cached. Whether it originates from Elastic or a
    local file, everything downstream operates on the same real ``DataFrame``,
    so the agent's exploration, statistics, and model training are all genuine.
    """

    def __init__(self, settings: Settings, elastic_client: Any | None = None) -> None:
        self._settings = settings
        self._elastic = elastic_client
        self._frame: pd.DataFrame | None = None
        self._profile: DataProfile | None = None
        self._mode: str = "local"
        self._source: str = settings.data_csv_path

    # ── Loading ─────────────────────────────────────────────────────────
    async def frame(self) -> pd.DataFrame:
        """Return the cached real DataFrame, loading it on first use."""
        if self._frame is None:
            self._frame = await self._load()
        return self._frame

    async def _load(self) -> pd.DataFrame:
        # A user-supplied local dataset always wins over the configured index.
        prefer_local = getattr(self._settings, "prefer_local_data", False)
        # Prefer live Elastic data when a client is available and enabled.
        if (
            not prefer_local
            and self._elastic is not None
            and getattr(self._elastic, "enabled", False)
        ):
            df = await self._load_from_elastic()
            if df is not None and not df.empty:
                self._mode = "live"
                self._source = self._settings.elastic_index
                return _coerce_dtypes(df)
            logger.info("Elastic returned no rows; falling back to local dataset.")
        df = self._load_from_file()
        self._mode = "local"
        self._source = self._settings.data_csv_path
        return _coerce_dtypes(df)

    async def _load_from_elastic(self) -> pd.DataFrame | None:
        try:
            rows = await self._elastic.fetch_rows(size=50_000)
            if not rows:
                return None
            return pd.DataFrame(rows)
        except Exception as exc:  # noqa: BLE001 - degrade to local data
            logger.warning("Elastic load failed (%s); using local dataset.", exc)
            return None

    def _load_from_file(self) -> pd.DataFrame:
        path = Path(self._settings.data_csv_path)
        if not path.exists():
            raise FileNotFoundError(
                f"No data source available: Elastic is not configured and "
                f"DATA_CSV_PATH ('{path}') does not exist. Point DATA_CSV_PATH "
                f"at a CSV or Parquet file to analyse."
            )
        if path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        return pd.read_csv(path)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def source(self) -> str:
        return self._source

    # ── Profiling ───────────────────────────────────────────────────────
    async def profile(self) -> DataProfile:
        """Profile every column of the real dataset (cached)."""
        if self._profile is None:
            df = await self.frame()
            cols = [profile_column(df[name], name) for name in df.columns]
            self._profile = DataProfile(
                source=self._source,
                mode=self._mode,
                row_count=int(len(df)),
                columns=cols,
            )
        return self._profile

    # ── Real aggregation (the Explore step) ─────────────────────────────
    async def aggregate(
        self, dimension: str, metric: str, agg: str = "mean"
    ) -> dict[str, Any]:
        """Group ``metric`` by ``dimension`` using a real pandas aggregation.

        Mirrors an Elasticsearch terms+metric aggregation, but computed on the
        real frame so the buckets reflect genuine data.
        """
        df = await self.frame()
        if dimension not in df.columns or metric not in df.columns:
            return {"buckets": [], "dimension": dimension, "metric": metric}
        grouped = df.groupby(dimension, dropna=True)[metric]
        agg_series = getattr(grouped, agg)()
        counts = grouped.count()
        buckets = [
            {
                "key": _jsonable(key),
                "doc_count": int(counts.loc[key]),
                "metric": {"value": float(value)},
            }
            for key, value in agg_series.sort_values(ascending=False).items()
            if pd.notna(value)
        ]
        return {"buckets": buckets, "dimension": dimension, "metric": metric}

    # ── Real association ranking (powers anomaly discovery + reasoning) ──
    async def rank_associations(
        self, target: str, top: int = 8
    ) -> list[dict[str, Any]]:
        """Rank features by their genuine univariate association with ``target``.

        Uses a real, cheap statistic per feature type (correlation for numeric,
        Cramér's V / mean-gap for categorical) so the agent can surface the
        strongest *real* signals before forming a hypothesis. Returns a list of
        ``{feature, strength, direction, kind}`` sorted strongest-first.
        """
        df = await self.frame()
        prof = await self.profile()
        target_col = prof.column(target)
        if target_col is None or target not in df.columns:
            return []

        results: list[dict[str, Any]] = []
        y = df[target]
        y_numeric = pd.api.types.is_numeric_dtype(y)

        for col in prof.candidate_features(target):
            name = col.name
            series = df[name]
            try:
                if col.role == ROLE_NUMERIC and y_numeric and target_col.role in (
                    ROLE_NUMERIC,
                ):
                    strength, direction = _abs_correlation(series, y)
                    kind = "correlation"
                elif col.role == ROLE_NUMERIC:
                    # numeric feature vs categorical/binary target → mean gap
                    strength, direction = _numeric_vs_group(df, name, target)
                    kind = "mean_gap"
                else:
                    # categorical feature vs any target → Cramér's V (binned y)
                    strength = _cramers_v(series, _as_categorical(y))
                    direction = "association"
                    kind = "cramers_v"
            except Exception:  # noqa: BLE001 - a single column never breaks ranking
                continue
            if strength is None or np.isnan(strength):
                continue
            results.append(
                {
                    "feature": name,
                    "strength": round(float(strength), 4),
                    "direction": direction,
                    "kind": kind,
                    "role": col.role,
                }
            )

        results.sort(key=lambda r: r["strength"], reverse=True)
        return results[:top]

    # ── Real feature/target samples for validation + charts ─────────────
    async def column_values(self, *names: str) -> dict[str, list[Any]]:
        """Return real, row-aligned values for the named columns (dropna joined)."""
        df = await self.frame()
        present = [n for n in names if n in df.columns]
        subset = df[present].dropna()
        return {n: subset[n].tolist() for n in present}

    async def feature_target_points(
        self, feature: str, target: str, bins: int = 12
    ) -> dict[str, Any]:
        """Bucket ``feature`` and return the mean ``target`` per bucket.

        Drives the dashboard's dynamic insight chart with real data: for a
        numeric feature, returns mean-target-per-bin; for a categorical feature,
        returns mean-target-per-category.
        """
        df = await self.frame()
        if feature not in df.columns or target not in df.columns:
            return {"points": [], "mode": self._mode}
        sub = df[[feature, target]].dropna()
        if sub.empty:
            return {"points": [], "mode": self._mode}

        y = sub[target]
        if not pd.api.types.is_numeric_dtype(y):
            y = _as_binary_numeric(y)
        sub = sub.assign(_y=y.to_numpy())

        points: list[dict[str, Any]] = []
        if pd.api.types.is_numeric_dtype(sub[feature]):
            try:
                cats = pd.qcut(sub[feature], q=min(bins, max(sub[feature].nunique(), 1)),
                               duplicates="drop")
            except (ValueError, IndexError):
                cats = pd.cut(sub[feature], bins=min(bins, 10))
            grouped = sub.groupby(cats, observed=True)
            for interval, g in grouped:
                if len(g) == 0:
                    continue
                center = float(getattr(interval, "mid", float(g[feature].mean())))
                points.append(
                    {"x": round(center, 2), "y": round(float(g["_y"].mean()), 4),
                     "count": int(len(g))}
                )
        else:
            grouped = sub.groupby(feature, observed=True)
            for key, g in grouped:
                points.append(
                    {"x": _jsonable(key), "y": round(float(g["_y"].mean()), 4),
                     "count": int(len(g))}
                )
        return {"points": points, "mode": self._mode}


    # ── Dataset summary (powers the UI's dataset card + suggestions) ────
    async def summary(self) -> dict[str, Any]:
        """Return a domain-agnostic snapshot of the loaded dataset."""
        prof = await self.profile()
        targets = prof.candidate_targets()
        return {
            "source": prof.source,
            "mode": prof.mode,
            "row_count": prof.row_count,
            "n_columns": len(prof.columns),
            "target": targets[0].name if targets else None,
            "candidate_targets": [t.name for t in targets[:6]],
            "columns": [{"name": c.name, "role": c.role} for c in prof.columns],
            "suggestions": suggest_questions(prof),
        }


def suggest_questions(profile: "DataProfile") -> list[str]:
    """Generate domain-agnostic mission prompts from the actual columns.

    Works for *any* dataset — churn, housing, sensors, sales — because the
    questions are phrased around whatever real targets the profiler found,
    never a hard-coded domain.
    """
    out: list[str] = []
    for col in profile.candidate_targets()[:3]:
        t = col.name
        if col.role == ROLE_NUMERIC:
            out.append(f"What features best explain {t}?")
        else:
            out.append(f"What most strongly drives {t}?")
    if not out:
        out.append("What is the strongest relationship in this dataset?")
    out.append("Surface the single biggest anomaly in this dataset.")
    # de-duplicate while preserving order, cap at 4
    seen, unique = set(), []
    for q in out:
        if q not in seen:
            seen.add(q); unique.append(q)
    return unique[:4]


# ── Module-level helpers ────────────────────────────────────────────────
def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort dtype coercion: parse obvious datetime columns."""
    for col in df.columns:
        if df[col].dtype == object:
            lowered = str(col).lower()
            if "date" in lowered or "time" in lowered or lowered.endswith("_at"):
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.8:
                    df[col] = parsed
    return df


def _as_categorical(y: pd.Series) -> pd.Series:
    """Bin a numeric target into <=10 quantile groups so it can pair with a
    categorical feature; pass categorical targets through unchanged."""
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
        try:
            return pd.qcut(y, q=5, duplicates="drop").astype(str)
        except ValueError:
            return y.astype(str)
    return y.astype(str)


def _as_binary_numeric(y: pd.Series) -> pd.Series:
    """Map a 2-class categorical target to 0/1 by its alphabetical order."""
    classes = sorted(y.dropna().unique().tolist(), key=str)
    if len(classes) <= 2 and classes:
        positive = classes[-1]
        return (y == positive).astype(int)
    # Many classes: encode the most frequent as the positive event.
    top = y.value_counts().idxmax()
    return (y == top).astype(int)


def _abs_correlation(x: pd.Series, y: pd.Series) -> tuple[float, str]:
    joined = pd.concat([x, y], axis=1).dropna()
    if len(joined) < 3:
        return float("nan"), "n/a"
    r = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
    direction = "positive" if r >= 0 else "negative"
    return abs(r), direction


def _numeric_vs_group(df: pd.DataFrame, feature: str, target: str) -> tuple[float, str]:
    """Strength of a numeric feature across target groups, as a normalised gap."""
    sub = df[[feature, target]].dropna()
    if sub.empty:
        return float("nan"), "n/a"
    means = sub.groupby(target, observed=True)[feature].mean()
    if means.empty or means.max() == means.min():
        return 0.0, "flat"
    spread = float(sub[feature].std()) or 1.0
    gap = float(means.max() - means.min()) / spread  # standardised gap
    hi_group = _jsonable(means.idxmax())
    return gap, f"higher when {target}={hi_group}"


def _cramers_v(x: pd.Series, y: pd.Series) -> float | None:
    """Cramér's V association between two categorical series (real, bias-free)."""
    from scipy.stats import chi2_contingency

    table = pd.crosstab(x.astype(str), y.astype(str))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None
    chi2 = chi2_contingency(table.to_numpy())[0]
    n = table.to_numpy().sum()
    if n == 0:
        return None
    phi2 = chi2 / n
    r, k = table.shape
    # Bias correction (Bergsma & Wicher).
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return None
    return float(np.sqrt(phi2corr / denom))
