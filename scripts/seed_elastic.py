"""Seed a synthetic telemetry index into the live Elastic cluster.

Creates the index named by ``ELASTIC_INDEX`` (default ``prod_db.api_logs_daily``)
with an explicit mapping, then bulk-inserts synthetic user rows whose
``latency_ms`` / ``user_status`` / ``churn_30_days`` fields encode a real
(latency → churn) association — so the agent's Explore + Validate steps operate
on genuine, queryable data.

Usage:
    PYTHONPATH=. python scripts/seed_elastic.py [--rows 50000] [--recreate]

This WRITES to your Elastic cluster. It is idempotent on the index name: pass
``--recreate`` to drop and rebuild the index.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from elasticsearch import Elasticsearch, helpers

from app.config import get_settings

MAPPING = {
    "mappings": {
        "properties": {
            "user_id": {"type": "keyword"},
            "user_status": {"type": "keyword"},
            "latency_ms": {"type": "float"},
            "api_calls_count": {"type": "integer"},
            "onboarding_step": {"type": "integer"},
            "churn_30_days": {"type": "integer"},
        }
    }
}


def build_client(settings) -> Elasticsearch:
    kwargs: dict = {"api_key": settings.elastic_api_key}
    if settings.elastic_cloud_id:
        kwargs["cloud_id"] = settings.elastic_cloud_id
    else:
        kwargs["hosts"] = [settings.elastic_url]
    return Elasticsearch(**kwargs)


def generate_rows(n: int, seed: int = 42):
    """Yield synthetic telemetry docs with a genuine latency→churn effect."""
    rng = np.random.default_rng(seed)
    latency = rng.normal(loc=340, scale=120, size=n).clip(min=20)
    high = latency > 450
    # Subtle but real: high-latency users churn somewhat more often.
    churn_prob = np.where(high, 0.205, 0.155)
    churn = (rng.random(n) < churn_prob).astype(int)
    api_calls = rng.integers(5, 500, size=n)
    onboarding = rng.integers(1, 6, size=n)

    for i in range(n):
        yield {
            "user_id": f"user_{i:06d}",
            "user_status": "churned" if churn[i] else "retained",
            "latency_ms": round(float(latency[i]), 2),
            "api_calls_count": int(api_calls[i]),
            "onboarding_step": int(onboarding[i]),
            "churn_30_days": int(churn[i]),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    if not (settings.elastic_api_key and (settings.elastic_cloud_id or settings.elastic_url)):
        print("Elastic is not configured in .env; nothing to seed.")
        return 1

    es = build_client(settings)
    index = settings.elastic_index

    if args.recreate and es.indices.exists(index=index):
        print(f"Dropping existing index '{index}'...")
        es.indices.delete(index=index)

    if not es.indices.exists(index=index):
        print(f"Creating index '{index}'...")
        es.indices.create(index=index, body=MAPPING)
    else:
        print(f"Index '{index}' already exists; appending {args.rows:,} rows.")

    print(f"Bulk-inserting {args.rows:,} synthetic rows...")
    actions = (
        {"_index": index, "_source": doc} for doc in generate_rows(args.rows)
    )
    success, errors = helpers.bulk(es, actions, chunk_size=2000, request_timeout=120)
    es.indices.refresh(index=index)

    count = es.count(index=index)["count"]
    print(f"Done. Inserted={success}, errors={len(errors) if errors else 0}, total_docs={count:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
