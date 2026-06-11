"""Phase 6 — Live data seeding.

One-off loader that bulk-inserts the Kaggle *SaaS Customer Churn Prediction*
dataset into the live Elasticsearch deployment, so Epiphany's MCP schema
discovery (Phase 3) can autonomously adapt to the new structure.

Dataset:
    https://www.kaggle.com/datasets/suhanigupta04/saas-customer-churn-prediction-dataset
    Download ``train.csv`` and place it at ``<project root>/data/train.csv``.

Usage:
    # 1) Make sure your .env has live Elastic creds and FORCE_SIMULATION=false
    # 2) Run from the project root:
    PYTHONPATH=. python scripts/seed_live_data.py
    # Optional flags:
    PYTHONPATH=. python scripts/seed_live_data.py --csv data/train.csv \
        --index prod_db.saas_telemetry --recreate

This writes to the SAME index the agent reads (``settings.elastic_index``), so
once seeding completes the next live cycle investigates real data.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

# Resolve the project root so the script works from any CWD.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402

DEFAULT_CSV = PROJECT_ROOT / "data" / "train.csv"


def _build_client(settings: Any):
    """Construct a synchronous Elasticsearch client from app settings."""
    try:
        from elasticsearch import Elasticsearch
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            "The 'elasticsearch' package is required. Install it with:\n"
            "    pip install elasticsearch"
        ) from exc

    if not settings.elastic_api_key or not (
        settings.elastic_cloud_id or settings.elastic_url
    ):
        raise SystemExit(
            "Elastic is not configured. Set ELASTIC_API_KEY and one of "
            "ELASTIC_CLOUD_ID / ELASTIC_URL in your .env before seeding."
        )

    kwargs: dict[str, Any] = {"api_key": settings.elastic_api_key}
    if settings.elastic_cloud_id:
        kwargs["cloud_id"] = settings.elastic_cloud_id
    else:
        kwargs["hosts"] = [settings.elastic_url]
    return Elasticsearch(**kwargs)


def _clean_record(row: dict[str, Any]) -> dict[str, Any]:
    """Drop NaN values so Elasticsearch infers clean field mappings."""
    clean: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float) and math.isnan(value):
            continue
        clean[str(key)] = value
    return clean


def _actions(df: pd.DataFrame, index: str) -> Iterator[dict[str, Any]]:
    """Yield bulk index actions for every DataFrame row."""
    for i, row in enumerate(df.to_dict(orient="records")):
        yield {"_index": index, "_id": i, "_source": _clean_record(row)}


def seed(csv_path: Path, index: str, recreate: bool) -> None:
    """Read the CSV and bulk-load it into the live index."""
    from elasticsearch import helpers

    if not csv_path.exists():
        raise SystemExit(
            f"Dataset not found at {csv_path}.\n"
            "Download train.csv from the Kaggle SaaS Customer Churn dataset and "
            f"place it at {csv_path}."
        )

    settings = get_settings()
    client = _build_client(settings)

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    print(f"Loaded {len(df):,} rows x {len(df.columns)} columns from {csv_path.name}")
    print(f"Columns: {', '.join(df.columns)}")

    if recreate and client.indices.exists(index=index):
        print(f"Deleting existing index '{index}'...")
        client.indices.delete(index=index)

    if not client.indices.exists(index=index):
        print(f"Creating index '{index}' (dynamic mapping)...")
        client.indices.create(index=index)

    print(f"Bulk-inserting {len(df):,} documents into '{index}'...")
    success, errors = helpers.bulk(
        client, _actions(df, index), chunk_size=1000, raise_on_error=False
    )
    client.indices.refresh(index=index)

    count = client.count(index=index)["count"]
    print(f"Done. Indexed {success:,} docs (errors: {len(errors)}).")
    print(f"Index '{index}' now holds {count:,} documents.")
    if errors:
        print("First error sample:", errors[0])
    client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed live Elastic with churn data.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to CSV.")
    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="Target index (defaults to settings.elastic_index).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the index before loading.",
    )
    args = parser.parse_args()

    index = args.index or get_settings().elastic_index
    seed(args.csv, index, args.recreate)


if __name__ == "__main__":
    main()
