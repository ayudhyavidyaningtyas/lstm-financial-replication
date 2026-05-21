#!/usr/bin/env python3
"""Build a documented S&P 500 adjusted-price panel.

This improves on a bare price download by saving ticker metadata, failed
downloads, missing-data diagnostics, and a reproducible metadata file. The
default ticker universe is the current S&P 500; for a stricter academic study,
pass a point-in-time constituents file with ``--constituents-csv``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_START = "2000-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DEFAULT_GITHUB_URL = (
    "https://raw.githubusercontent.com/datasets/"
    "s-and-p-500-companies/main/data/constituents.csv"
)
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


@dataclass
class DownloadConfig:
    start: str
    end: str
    chunk_size: int
    min_obs_fraction: float
    output_dir: str
    constituents_csv: str | None
    ticker_column: str
    source: str


def yahoo_symbol(symbol: str) -> str:
    """Convert index-style share class tickers to Yahoo format."""
    return str(symbol).strip().replace(".", "-")


def index_symbol(yahoo_ticker: str) -> str:
    """Convert Yahoo share class tickers back to index-style notation."""
    return str(yahoo_ticker).strip().replace("-", ".")


def read_constituents_from_csv(path: Path, ticker_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if ticker_column not in df.columns:
        raise ValueError(
            f"Ticker column '{ticker_column}' not found in {path}. "
            f"Available columns: {', '.join(df.columns)}"
        )
    out = df.copy()
    out["Symbol"] = out[ticker_column].astype(str)
    out["Security"] = out.get("Security", out["Symbol"])
    out["GICS Sector"] = out.get("GICS Sector", "")
    out["GICS Sub-Industry"] = out.get("GICS Sub-Industry", "")
    out["source"] = str(path)
    return out


def read_current_constituents(source: str) -> pd.DataFrame:
    """Read current S&P 500 constituents from GitHub or Wikipedia."""
    errors: list[str] = []

    if source in {"auto", "github"}:
        try:
            response = requests.get(DEFAULT_GITHUB_URL, headers=HEADERS, timeout=30)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            df["source"] = DEFAULT_GITHUB_URL
            return df
        except Exception as exc:  # pragma: no cover - depends on network.
            errors.append(f"GitHub failed: {exc}")
            if source == "github":
                raise

    if source in {"auto", "wikipedia"}:
        try:
            response = requests.get(DEFAULT_WIKI_URL, headers=HEADERS, timeout=30)
            response.raise_for_status()
            df = pd.read_html(StringIO(response.text))[0]
            df["source"] = DEFAULT_WIKI_URL
            return df
        except Exception as exc:  # pragma: no cover - depends on network.
            errors.append(f"Wikipedia failed: {exc}")
            raise RuntimeError("; ".join(errors)) from exc

    raise ValueError("source must be one of: auto, github, wikipedia")


def load_constituents(args: argparse.Namespace) -> pd.DataFrame:
    if args.constituents_csv:
        df = read_constituents_from_csv(Path(args.constituents_csv), args.ticker_column)
    else:
        df = read_current_constituents(args.source)

    if "Symbol" not in df.columns:
        raise ValueError("Constituent table must contain a 'Symbol' column")

    df = df.copy()
    df["index_symbol"] = df["Symbol"].astype(str).str.strip()
    df["ticker"] = df["index_symbol"].map(yahoo_symbol)
    df = df[df["ticker"].ne("")].drop_duplicates("ticker").sort_values("ticker")

    ordered_columns = [
        "ticker",
        "index_symbol",
        "Security",
        "GICS Sector",
        "GICS Sub-Industry",
        "Date added",
        "source",
    ]
    for column in ordered_columns:
        if column not in df.columns:
            df[column] = ""
    return df[ordered_columns].reset_index(drop=True)


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def extract_adjusted_close(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Handle yfinance's different return layouts for one vs many tickers."""
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"]
        elif "Adj Close" in data.columns.get_level_values(0):
            close = data["Adj Close"]
        else:
            raise ValueError(f"No Close/Adj Close columns found: {data.columns}")
    else:
        column = "Close" if "Close" in data.columns else "Adj Close"
        close = data[[column]].copy()
        close.columns = tickers[:1]

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    close = close.copy()
    close.columns = [str(col) for col in close.columns]
    return close


def download_prices(
    tickers: list[str],
    *,
    start: str,
    end: str,
    chunk_size: int,
    pause: float,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frames: list[pd.DataFrame] = []
    logs: list[dict[str, object]] = []

    for chunk_no, ticker_chunk in enumerate(chunks(tickers, chunk_size), start=1):
        first = (chunk_no - 1) * chunk_size + 1
        last = first + len(ticker_chunk) - 1
        print(f"Downloading {first}-{last} of {len(tickers)}: {', '.join(ticker_chunk[:4])}...")
        record: dict[str, object] = {
            "chunk": chunk_no,
            "tickers": ticker_chunk,
            "status": "ok",
            "error": "",
        }
        try:
            data = yf.download(
                ticker_chunk,
                start=start,
                end=end,
                auto_adjust=True,
                actions=False,
                group_by="column",
                progress=False,
                threads=True,
            )
            close = extract_adjusted_close(data, ticker_chunk)
            frames.append(close)
            all_nan = sorted([col for col in close.columns if close[col].isna().all()])
            record["all_nan_tickers"] = all_nan
            record["rows"] = int(len(close))
            record["columns"] = int(close.shape[1])
            if all_nan:
                record["status"] = "partial"
        except Exception as exc:  # pragma: no cover - depends on network.
            record["status"] = "failed"
            record["error"] = repr(exc)
            print(f"  failed: {exc}")
        logs.append(record)
        if pause > 0:
            time.sleep(pause)

    if not frames:
        raise RuntimeError("No price data downloaded")

    panel = pd.concat(frames, axis=1)
    panel = panel.loc[:, ~panel.columns.duplicated()].sort_index()
    panel = panel.reindex(sorted(panel.columns), axis=1)
    panel.index.name = "Date"
    return panel, logs


def quality_report(panel: pd.DataFrame) -> pd.DataFrame:
    returns = panel.pct_change(fill_method=None)
    rows = []
    for ticker in panel.columns:
        series = panel[ticker]
        valid = series.dropna()
        ret = returns[ticker].dropna()
        rows.append(
            {
                "ticker": ticker,
                "observations": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "missing_fraction": float(series.isna().mean()),
                "first_valid_date": valid.index.min().date().isoformat() if len(valid) else "",
                "last_valid_date": valid.index.max().date().isoformat() if len(valid) else "",
                "non_positive_prices": int((valid <= 0).sum()),
                "zero_return_days": int((ret == 0).sum()),
                "return_abs_gt_50pct": int((ret.abs() > 0.5).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_fraction", "ticker"]).reset_index(drop=True)


def clean_panel(
    panel: pd.DataFrame,
    *,
    min_obs_fraction: float,
    keep_all: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    report = quality_report(panel)
    if keep_all:
        cleaned = panel
    else:
        min_obs = int(np.ceil(min_obs_fraction * len(panel)))
        keep = report.loc[report["observations"] >= min_obs, "ticker"].tolist()
        cleaned = panel[keep]
    return cleaned.sort_index(), report


def write_outputs(
    output_dir: Path,
    panel: pd.DataFrame,
    raw_panel: pd.DataFrame,
    constituents: pd.DataFrame,
    report: pd.DataFrame,
    download_log: list[dict[str, object]],
    config: DownloadConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    panel.to_csv(output_dir / "sp500_prices.csv")
    raw_panel.to_csv(output_dir / "sp500_prices_raw_unfiltered.csv")
    constituents.to_csv(output_dir / "sp500_constituents.csv", index=False)
    report.to_csv(output_dir / "sp500_price_quality_report.csv", index=False)

    failed = []
    for row in download_log:
        if row.get("status") == "failed":
            failed.extend(row.get("tickers", []))
        failed.extend(row.get("all_nan_tickers", []))

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "n_constituents": int(len(constituents)),
        "raw_shape": list(raw_panel.shape),
        "clean_shape": list(panel.shape),
        "date_min": panel.index.min().date().isoformat() if len(panel) else "",
        "date_max": panel.index.max().date().isoformat() if len(panel) else "",
        "failed_or_empty_tickers": sorted(set(map(str, failed))),
        "download_log": download_log,
        "notes": [
            "Prices are Yahoo Finance adjusted Close via yfinance auto_adjust=True.",
            "Missing prices are preserved; the modelling pipeline filters invalid windows.",
            "Default constituents are current S&P 500 names and therefore may introduce survivorship bias.",
            "For a stricter replication, provide a point-in-time historical constituents file.",
        ],
    }
    (output_dir / "sp500_download_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--pause", type=float, default=0.25, help="Seconds to pause between chunks")
    parser.add_argument("--min-obs-fraction", type=float, default=0.20)
    parser.add_argument("--keep-all", action="store_true", help="Keep tickers even with sparse history")
    parser.add_argument("--constituents-csv", default=None, help="Optional custom constituents CSV")
    parser.add_argument("--ticker-column", default="Symbol", help="Ticker column in custom constituents CSV")
    parser.add_argument("--source", choices=["auto", "github", "wikipedia"], default="auto")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()

    constituents = load_constituents(args)
    tickers = constituents["ticker"].tolist()
    print(f"{len(tickers)} tickers loaded.")

    raw_panel, download_log = download_prices(
        tickers,
        start=args.start,
        end=args.end,
        chunk_size=args.chunk_size,
        pause=args.pause,
    )
    panel, report = clean_panel(
        raw_panel,
        min_obs_fraction=args.min_obs_fraction,
        keep_all=args.keep_all,
    )

    config = DownloadConfig(
        start=args.start,
        end=args.end,
        chunk_size=args.chunk_size,
        min_obs_fraction=args.min_obs_fraction,
        output_dir=str(output_dir),
        constituents_csv=args.constituents_csv,
        ticker_column=args.ticker_column,
        source=args.source,
    )
    write_outputs(output_dir, panel, raw_panel, constituents, report, download_log, config)

    print(
        f"\nSaved {output_dir / 'sp500_prices.csv'}: "
        f"{panel.shape[0]} trading days x {panel.shape[1]} tickers."
    )
    print(f"Quality report: {output_dir / 'sp500_price_quality_report.csv'}")
    print(f"Metadata: {output_dir / 'sp500_download_metadata.json'}")


if __name__ == "__main__":
    main()
