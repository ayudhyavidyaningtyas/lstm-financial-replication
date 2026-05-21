#!/usr/bin/env python3
"""Build a historical S&P 500 union ticker list.

This reads the public fja05680/sp500 historical components file and writes:

1. A single-column historical-union constituents CSV that can be passed to
   download_sp500_prices.py.
2. A long-form membership snapshot CSV (date, Symbol) that model scripts can
   use as a point-in-time membership mask.

Important limitation:
    The output is a historical union, not a daily membership mask. It reduces
    current-constituent survivorship bias by adding removed historical members
    back into the download universe, but downstream models still treat any stock
    with price data as eligible unless a true point-in-time mask is added.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


PIT_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes.csv"
)
REMOVAL_SUFFIX_RE = re.compile(r"-\d{6}$")


def to_yahoo_ticker(token: str) -> str:
    """Normalize fja05680 ticker tokens to Yahoo Finance format."""
    bare = REMOVAL_SUFFIX_RE.sub("", str(token).strip())
    return bare.replace(".", "-")


def parse_ticker_string(tickers_string: str) -> list[str]:
    tickers = []
    for token in str(tickers_string).split(","):
        symbol = to_yahoo_ticker(token)
        if symbol:
            tickers.append(symbol)
    return tickers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2000-01-01", help="Study window start date")
    parser.add_argument("--end", default="2024-12-31", help="Study window end date")
    parser.add_argument("--output", default="data/historical_constituents.csv")
    parser.add_argument("--cache", default="data/sp500_pit_raw.csv")
    parser.add_argument("--metadata", default="data/historical_constituents_metadata.json")
    parser.add_argument("--snapshots", default="data/sp500_membership_snapshots.csv")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        print(f"Downloading PIT components file to {cache_path} ...")
        response = requests.get(PIT_URL, timeout=60)
        response.raise_for_status()
        cache_path.write_bytes(response.content)
    else:
        print(f"Using cached PIT file at {cache_path}")

    raw = pd.read_csv(cache_path)
    if not {"date", "tickers"}.issubset(raw.columns):
        raise ValueError(f"Expected columns date,tickers; got {list(raw.columns)}")

    raw["date"] = pd.to_datetime(raw["date"])
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    window = raw.loc[(raw["date"] >= start) & (raw["date"] <= end)].copy()
    if window.empty:
        raise SystemExit(f"No PIT rows found from {args.start} to {args.end}")

    universe: set[str] = set()
    row_counts = []
    snapshot_rows = []
    for row in window.itertuples(index=False):
        tickers = sorted(set(parse_ticker_string(row.tickers)))
        universe.update(tickers)
        row_date = row.date.date().isoformat()
        row_counts.append({"date": row_date, "n_tickers": len(tickers)})
        snapshot_rows.extend({"date": row_date, "Symbol": symbol} for symbol in tickers)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Symbol": sorted(universe)}).to_csv(output_path, index=False)

    snapshots_path = Path(args.snapshots)
    snapshots_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(snapshot_rows).to_csv(snapshots_path, index=False)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": PIT_URL,
        "cache": str(cache_path),
        "start": args.start,
        "end": args.end,
        "rows_in_window": int(len(window)),
        "first_pit_row": window["date"].min().date().isoformat(),
        "last_pit_row": window["date"].max().date().isoformat(),
        "unique_union_tickers": int(len(universe)),
        "output": str(output_path),
        "snapshots": str(snapshots_path),
        "row_counts": row_counts,
        "limitations": [
            "This file is a historical union of members, not a daily point-in-time membership mask.",
            "Removed/delisted tickers are included when present in the source, but Yahoo may not provide complete delisting returns.",
            "The snapshot file can be used as a daily membership mask by model scripts via --membership-csv.",
        ],
    }
    metadata_path = Path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        f"PIT rows in window: {len(window):,} "
        f"({window['date'].min().date()} to {window['date'].max().date()})"
    )
    print(f"Unique historical-union tickers: {len(universe):,}")
    print(f"Wrote {output_path}")
    print(f"Wrote {snapshots_path}")
    print(f"Wrote {metadata_path}")
    print("\nNext:")
    print("  python3 src/download_sp500_prices.py \\")
    print(f"    --constituents-csv {output_path} \\")
    print("    --ticker-column Symbol \\")
    print(f"    --start {args.start} --end {args.end} \\")
    print("    --output-dir data")


if __name__ == "__main__":
    main()
