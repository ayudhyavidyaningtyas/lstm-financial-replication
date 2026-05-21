#!/usr/bin/env python3
"""Replicate Fischer and Krauss (2018) on the supplied S&P 500 prices.

The original paper trains a separate LSTM on rolling 750-day training windows,
then trades the next 250 days by going long the stocks with the highest
probability of beating the cross-sectional median and short the lowest-ranked
stocks. The command-line defaults are paper-style; pass ``--quick`` for a small
development run.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = str(PROJECT_ROOT / "data" / "sp500_prices.csv")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
SUBPERIOD_PRESETS = {
    "decades": [
        ("2000_2009", "2000-01-01", "2009-12-31"),
        ("2010_2019", "2010-01-01", "2019-12-31"),
        ("2020_2024", "2020-01-01", "2024-12-31"),
    ]
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(-drawdown.min())


def load_price_panel(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Price file not found: {path}\n"
            "Run `python3 src/download_sp500_prices.py --output-dir data` first, "
            "or pass `--csv /path/to/sp500_prices.csv`."
        )
    return pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()


def load_membership_mask(
    membership_csv: str | None,
    dates: pd.DatetimeIndex,
    tickers: list[str],
) -> np.ndarray | None:
    """Return a dates x tickers membership mask from long snapshot data."""
    if not membership_csv:
        return None

    path = Path(membership_csv).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Membership file not found: {path}")

    membership = pd.read_csv(path, parse_dates=["date"])
    if not {"date", "Symbol"}.issubset(membership.columns):
        raise ValueError("Membership CSV must contain columns: date, Symbol")

    membership["Symbol"] = membership["Symbol"].astype(str).str.replace(".", "-", regex=False)
    ticker_to_idx = {ticker: i for i, ticker in enumerate(tickers)}
    snapshots: list[tuple[pd.Timestamp, np.ndarray]] = []
    for snapshot_date, group in membership.groupby("date", sort=True):
        row = np.zeros(len(tickers), dtype=bool)
        indices = [ticker_to_idx[symbol] for symbol in group["Symbol"] if symbol in ticker_to_idx]
        if indices:
            row[indices] = True
        snapshots.append((pd.Timestamp(snapshot_date), row))

    if not snapshots:
        raise ValueError("Membership CSV contains no usable membership rows")

    snapshot_dates = pd.DatetimeIndex([item[0] for item in snapshots])
    snapshot_values = [item[1] for item in snapshots]
    positions = snapshot_dates.searchsorted(dates, side="right") - 1
    mask = np.zeros((len(dates), len(tickers)), dtype=bool)
    for date_idx, pos in enumerate(positions):
        if pos >= 0:
            mask[date_idx] = snapshot_values[int(pos)]
    return mask


def compute_signal_medians(
    raw_returns: np.ndarray,
    membership_mask: np.ndarray | None,
) -> np.ndarray:
    """Compute median next-day return over stocks eligible on signal day t."""
    medians = np.full(raw_returns.shape[0], np.nan, dtype=np.float32)
    if membership_mask is None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            medians[:-1] = np.nanmedian(raw_returns[1:], axis=1)
        return medians

    for end_idx in range(raw_returns.shape[0] - 1):
        eligible = membership_mask[end_idx] & np.isfinite(raw_returns[end_idx + 1])
        if np.any(eligible):
            medians[end_idx] = float(np.nanmedian(raw_returns[end_idx + 1, eligible]))
    return medians


def chronological_train_val_indices(
    n_samples: int,
    validation_fraction: float,
    validation_order: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out the most recent observations for validation.

    `validation_order` should be a time-like integer/date array such as target
    date index. Whole dates are assigned to validation together when possible,
    which avoids leaking later calendar samples into model selection.
    """
    if n_samples <= 1:
        idx = np.arange(n_samples)
        return idx, idx[:0]

    if validation_order is None:
        ordered = np.arange(n_samples)
        split = max(1, int(n_samples * (1.0 - validation_fraction)))
        split = min(split, n_samples - 1)
        return ordered[:split], ordered[split:]

    order_values = np.asarray(validation_order)
    if len(order_values) != n_samples:
        raise ValueError("validation_order must have the same length as x/y")

    unique_values = np.unique(order_values)
    if len(unique_values) <= 1:
        ordered = np.argsort(order_values, kind="stable")
        split = max(1, int(n_samples * (1.0 - validation_fraction)))
        split = min(split, n_samples - 1)
        return ordered[:split], ordered[split:]

    n_val_dates = max(1, int(math.ceil(len(unique_values) * validation_fraction)))
    cutoff = unique_values[-n_val_dates]
    train_idx = np.flatnonzero(order_values < cutoff)
    val_idx = np.flatnonzero(order_values >= cutoff)

    if len(train_idx) == 0 or len(val_idx) == 0:
        ordered = np.argsort(order_values, kind="stable")
        split = max(1, int(n_samples * (1.0 - validation_fraction)))
        split = min(split, n_samples - 1)
        return ordered[:split], ordered[split:]
    return train_idx, val_idx


@dataclass
class LSTMSettings:
    seq_len: int = 60
    hidden_dim: int = 12
    learning_rate: float = 0.002
    batch_size: int = 128
    max_epochs: int = 30
    patience: int = 5
    dropout: float = 0.05
    validation_fraction: float = 0.2
    rmsprop_rho: float = 0.9
    gradient_clip: float = 5.0


class NumpyLSTMClassifier:
    """Small LSTM classifier with RMSprop, implemented only with NumPy.

    It is intentionally compact and dependency-light. It is not a replacement
    for Keras/PyTorch, but it mirrors the architecture used in the paper:
    one sequence feature, an LSTM layer, and a two-class softmax head.
    """

    def __init__(self, settings: LSTMSettings, seed: int = 42) -> None:
        self.settings = settings
        self.rng = np.random.default_rng(seed)
        self.input_dim = 1
        h = settings.hidden_dim
        scale = 1.0 / math.sqrt(h + self.input_dim)
        self.params: dict[str, np.ndarray] = {
            "Wf": self.rng.normal(0.0, scale, size=(self.input_dim + h, h)).astype(np.float32),
            "Wi": self.rng.normal(0.0, scale, size=(self.input_dim + h, h)).astype(np.float32),
            "Wg": self.rng.normal(0.0, scale, size=(self.input_dim + h, h)).astype(np.float32),
            "Wo": self.rng.normal(0.0, scale, size=(self.input_dim + h, h)).astype(np.float32),
            "bf": np.ones(h, dtype=np.float32),
            "bi": np.zeros(h, dtype=np.float32),
            "bg": np.zeros(h, dtype=np.float32),
            "bo": np.zeros(h, dtype=np.float32),
            "Wy": self.rng.normal(0.0, scale, size=(h, 2)).astype(np.float32),
            "by": np.zeros(2, dtype=np.float32),
        }
        self.rms_cache = {name: np.zeros_like(value) for name, value in self.params.items()}

    def _forward(
        self,
        x: np.ndarray,
        y: np.ndarray | None = None,
        *,
        training: bool = False,
        store_cache: bool = False,
    ) -> tuple[float | None, np.ndarray, dict[str, list[np.ndarray] | np.ndarray] | None]:
        x_used = x
        if training and self.settings.dropout > 0.0:
            keep = 1.0 - self.settings.dropout
            mask = (self.rng.random(x.shape) < keep).astype(np.float32) / keep
            x_used = x * mask

        batch_size, seq_len, _ = x_used.shape
        h_dim = self.settings.hidden_dim
        h = np.zeros((batch_size, h_dim), dtype=np.float32)
        c = np.zeros((batch_size, h_dim), dtype=np.float32)

        cache: dict[str, list[np.ndarray] | np.ndarray] | None = None
        if store_cache:
            cache = {key: [] for key in ("z", "f", "i", "g", "o", "c", "c_prev")}

        p = self.params
        for t in range(seq_len):
            z = np.concatenate([x_used[:, t, :], h], axis=1)
            c_prev = c
            f = sigmoid(z @ p["Wf"] + p["bf"])
            i = sigmoid(z @ p["Wi"] + p["bi"])
            g = np.tanh(z @ p["Wg"] + p["bg"])
            o = sigmoid(z @ p["Wo"] + p["bo"])
            c = f * c_prev + i * g
            h = o * np.tanh(c)
            if cache is not None:
                cache["z"].append(z)
                cache["f"].append(f)
                cache["i"].append(i)
                cache["g"].append(g)
                cache["o"].append(o)
                cache["c"].append(c)
                cache["c_prev"].append(c_prev)

        logits = h @ p["Wy"] + p["by"]
        probs = softmax(logits)
        loss = None
        if y is not None:
            loss = float(-np.mean(np.log(probs[np.arange(batch_size), y] + 1e-12)))
        if cache is not None:
            cache["h_last"] = h
        return loss, probs, cache

    def _backward(
        self,
        probs: np.ndarray,
        y: np.ndarray,
        cache: dict[str, list[np.ndarray] | np.ndarray],
    ) -> dict[str, np.ndarray]:
        p = self.params
        batch_size = y.shape[0]
        h_last = cache["h_last"]
        assert isinstance(h_last, np.ndarray)

        dy = probs.copy()
        dy[np.arange(batch_size), y] -= 1.0
        dy /= batch_size

        grads = {name: np.zeros_like(value) for name, value in p.items()}
        grads["Wy"] = h_last.T @ dy
        grads["by"] = dy.sum(axis=0)
        dh_next = dy @ p["Wy"].T
        dc_next = np.zeros_like(dh_next)

        for t in reversed(range(len(cache["z"]))):
            z = cache["z"][t]
            f = cache["f"][t]
            i = cache["i"][t]
            g = cache["g"][t]
            o = cache["o"][t]
            c = cache["c"][t]
            c_prev = cache["c_prev"][t]
            assert all(isinstance(arr, np.ndarray) for arr in (z, f, i, g, o, c, c_prev))

            tanh_c = np.tanh(c)
            do = dh_next * tanh_c
            dc = dh_next * o * (1.0 - tanh_c * tanh_c) + dc_next
            df = dc * c_prev
            di = dc * g
            dg = dc * i
            dc_next = dc * f

            da_f = df * f * (1.0 - f)
            da_i = di * i * (1.0 - i)
            da_g = dg * (1.0 - g * g)
            da_o = do * o * (1.0 - o)

            grads["Wf"] += z.T @ da_f
            grads["Wi"] += z.T @ da_i
            grads["Wg"] += z.T @ da_g
            grads["Wo"] += z.T @ da_o
            grads["bf"] += da_f.sum(axis=0)
            grads["bi"] += da_i.sum(axis=0)
            grads["bg"] += da_g.sum(axis=0)
            grads["bo"] += da_o.sum(axis=0)

            dz = da_f @ p["Wf"].T + da_i @ p["Wi"].T + da_g @ p["Wg"].T + da_o @ p["Wo"].T
            dh_next = dz[:, self.input_dim :]

        return self._clip_gradients(grads)

    def _clip_gradients(self, grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        norm = math.sqrt(float(sum(np.sum(g * g) for g in grads.values())))
        if norm > self.settings.gradient_clip:
            scale = self.settings.gradient_clip / (norm + 1e-12)
            grads = {name: grad * scale for name, grad in grads.items()}
        return grads

    def _apply_rmsprop(self, grads: dict[str, np.ndarray]) -> None:
        s = self.settings
        for name, grad in grads.items():
            self.rms_cache[name] = s.rmsprop_rho * self.rms_cache[name] + (1.0 - s.rmsprop_rho) * grad * grad
            self.params[name] -= s.learning_rate * grad / (np.sqrt(self.rms_cache[name]) + 1e-7)

    def _loss_in_batches(self, x: np.ndarray, y: np.ndarray) -> float:
        total_loss = 0.0
        total_n = 0
        for start in range(0, len(x), self.settings.batch_size):
            end = start + self.settings.batch_size
            loss, _, _ = self._forward(x[start:end], y[start:end], training=False, store_cache=False)
            if loss is not None:
                total_loss += loss * len(x[start:end])
                total_n += len(x[start:end])
        return total_loss / max(total_n, 1)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        validation_order: np.ndarray | None = None,
    ) -> list[dict[str, float]]:
        s = self.settings
        train_idx, val_idx = chronological_train_val_indices(
            len(x),
            s.validation_fraction,
            validation_order,
        )
        x_train, y_train = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]

        best_loss = math.inf
        best_params = {name: value.copy() for name, value in self.params.items()}
        bad_epochs = 0
        history: list[dict[str, float]] = []

        for epoch in range(1, s.max_epochs + 1):
            epoch_order = self.rng.permutation(len(x_train))
            train_loss_sum = 0.0
            train_n = 0
            for start in range(0, len(epoch_order), s.batch_size):
                idx = epoch_order[start : start + s.batch_size]
                xb, yb = x_train[idx], y_train[idx]
                loss, probs, cache = self._forward(xb, yb, training=True, store_cache=True)
                assert cache is not None and loss is not None
                grads = self._backward(probs, yb, cache)
                self._apply_rmsprop(grads)
                train_loss_sum += loss * len(xb)
                train_n += len(xb)

            train_loss = train_loss_sum / max(train_n, 1)
            val_loss = self._loss_in_batches(x_val, y_val) if len(x_val) else train_loss
            history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
            print(f"    epoch {epoch:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

            if val_loss + 1e-5 < best_loss:
                best_loss = val_loss
                best_params = {name: value.copy() for name, value in self.params.items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= s.patience:
                    break

        self.params = best_params
        return history

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        chunks = []
        for start in range(0, len(x), self.settings.batch_size):
            _, probs, _ = self._forward(x[start : start + self.settings.batch_size], store_cache=False)
            chunks.append(probs)
        return np.vstack(chunks)


def parse_k_values(values: Iterable[str]) -> list[int]:
    output: list[int] = []
    for value in values:
        for piece in str(value).split(","):
            piece = piece.strip()
            if piece:
                output.append(int(piece))
    return sorted(set(output))


def choose_period_starts(
    n_days: int,
    train_days: int,
    trade_days: int,
    periods: str,
    selection: str,
) -> list[int]:
    all_starts = list(range(0, n_days - train_days - trade_days + 1, trade_days))
    if periods.lower() == "all":
        return all_starts
    n_periods = int(periods)
    if selection == "recent":
        return all_starts[-n_periods:]
    return all_starts[:n_periods]


def parse_subperiods(value: str) -> list[tuple[str, str, str]]:
    if value.lower() in {"none", ""}:
        return []
    if value.lower() in SUBPERIOD_PRESETS:
        return SUBPERIOD_PRESETS[value.lower()]

    periods: list[tuple[str, str, str]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            label, start, end = item.split(":")
        except ValueError as exc:
            raise ValueError(
                "Subperiods must be 'decades' or comma-separated label:start:end entries, "
                "e.g. early:2000-01-01:2009-12-31,recent:2010-01-01:2024-12-31"
            ) from exc
        periods.append((label, start, end))
    return periods


def build_samples(
    raw_returns: np.ndarray,
    std_returns: np.ndarray,
    medians: np.ndarray,
    dates: pd.DatetimeIndex,
    tickers: list[str],
    *,
    first_end: int,
    last_end: int,
    seq_len: int,
    max_samples: int | None,
    rng: np.random.Generator,
    membership_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if last_end < first_end:
        raise ValueError("No sequence endpoints available for this split")

    end_indices = np.arange(first_end, last_end + 1)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    meta_parts: list[pd.DataFrame] = []

    for stock_idx, ticker in enumerate(tickers):
        window_start = first_end - seq_len + 1
        windows = np.lib.stride_tricks.sliding_window_view(
            std_returns[window_start : last_end + 1, stock_idx],
            seq_len,
        )
        target_indices = end_indices + 1
        next_returns = raw_returns[target_indices, stock_idx]
        target_medians = medians[end_indices]
        valid = np.isfinite(windows).all(axis=1) & np.isfinite(next_returns) & np.isfinite(target_medians)
        if membership_mask is not None:
            valid &= membership_mask[end_indices, stock_idx]
        if not np.any(valid):
            continue

        labels = (next_returns[valid] >= target_medians[valid]).astype(np.int64)
        valid_end_indices = end_indices[valid]
        valid_target_indices = target_indices[valid]
        valid_windows = windows[valid].astype(np.float32, copy=True)

        x_parts.append(valid_windows)
        y_parts.append(labels)
        meta_parts.append(
            pd.DataFrame(
                {
                    "target_date": dates[valid_target_indices],
                    "end_idx": valid_end_indices,
                    "target_idx": valid_target_indices,
                    "ticker": ticker,
                    "stock_idx": stock_idx,
                    "realized_return": next_returns[valid].astype(np.float32),
                    "label": labels,
                }
            )
        )

    if not x_parts:
        raise ValueError("No valid samples after filtering missing values")

    x = np.concatenate(x_parts, axis=0)[:, :, None]
    y = np.concatenate(y_parts, axis=0)
    meta = pd.concat(meta_parts, ignore_index=True)

    if max_samples is not None and max_samples > 0 and len(x) > max_samples:
        keep = rng.choice(len(x), size=max_samples, replace=False)
        x = x[keep]
        y = y[keep]
        meta = meta.iloc[keep].reset_index(drop=True)

    # Keep samples in calendar order. The model also receives target_idx
    # explicitly for validation, but this makes the ordering contract obvious.
    order = np.argsort(meta["target_idx"].to_numpy(), kind="stable")
    x = x[order]
    y = y[order]
    meta = meta.iloc[order].reset_index(drop=True)

    return x, y, meta


def add_reversal_score(meta: pd.DataFrame, raw_returns: np.ndarray, horizon: int = 5) -> pd.Series:
    scores = np.full(len(meta), np.nan, dtype=np.float32)
    for row_idx, row in enumerate(meta.itertuples(index=False)):
        end_idx = int(row.end_idx)
        stock_idx = int(row.stock_idx)
        start_idx = end_idx - horizon + 1
        if start_idx < 0:
            continue
        trailing = raw_returns[start_idx : end_idx + 1, stock_idx]
        if np.isfinite(trailing).all():
            cum_return = float(np.prod(1.0 + trailing) - 1.0)
            scores[row_idx] = -cum_return
    return pd.Series(scores, index=meta.index, name="reversal_score")


def evaluate_ranked_portfolios(
    meta: pd.DataFrame,
    scores: np.ndarray,
    k_values: list[int],
    method: str,
    half_turn_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = meta[["target_date", "ticker", "realized_return", "label"]].copy()
    work["score"] = scores
    work = work[np.isfinite(work["score"])].copy()
    daily_rows: list[dict[str, float | int | str | pd.Timestamp]] = []

    for date, group in work.groupby("target_date", sort=True):
        group = group.sort_values("score", ascending=False)
        returns = group["realized_return"].to_numpy(dtype=np.float64)
        labels = group["label"].to_numpy(dtype=np.int64)
        for k in k_values:
            if len(group) < 2 * k:
                continue
            long_idx = np.arange(k)
            short_idx = np.arange(len(group) - k, len(group))
            long_return = float(np.mean(returns[long_idx]))
            short_profit = float(-np.mean(returns[short_idx]))
            before_cost = long_return + short_profit
            after_cost = before_cost - 4.0 * half_turn_cost
            selected_correct = np.concatenate([labels[long_idx] == 1, labels[short_idx] == 0])
            daily_rows.append(
                {
                    "date": date,
                    "method": method,
                    "k": k,
                    "long_return": long_return,
                    "short_profit": short_profit,
                    "return_before_cost": before_cost,
                    "return_after_cost": after_cost,
                    "accuracy": float(np.mean(selected_correct)),
                    "n_ranked": int(len(group)),
                }
            )

    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        return daily, daily

    summary_rows = []
    for (summary_method, k), group in daily.groupby(["method", "k"], sort=True):
        before = group["return_before_cost"]
        after = group["return_after_cost"]
        summary_rows.append(
            {
                "method": summary_method,
                "k": int(k),
                "days": int(len(group)),
                "mean_return_before_cost": float(before.mean()),
                "mean_return_after_cost": float(after.mean()),
                "std_after_cost": float(after.std(ddof=1)),
                "sharpe_after_cost": float(after.mean() / after.std(ddof=1) * math.sqrt(252.0)),
                "accuracy": float(group["accuracy"].mean()),
                "long_mean": float(group["long_return"].mean()),
                "short_profit_mean": float(group["short_profit"].mean()),
                "max_drawdown_after_cost": max_drawdown(after),
                "positive_days_after_cost": float((after > 0.0).mean()),
            }
        )
    return daily, pd.DataFrame(summary_rows)


def selected_sequence_profile(
    x_trade: np.ndarray,
    meta: pd.DataFrame,
    scores: np.ndarray,
    k: int,
) -> pd.DataFrame:
    work = meta[["target_date"]].copy()
    work["score"] = scores
    work["row"] = np.arange(len(work))
    selected_top: list[int] = []
    selected_bottom: list[int] = []

    for _, group in work[np.isfinite(work["score"])].groupby("target_date", sort=True):
        if len(group) < 2 * k:
            continue
        ordered = group.sort_values("score", ascending=False)
        selected_top.extend(ordered.head(k)["row"].tolist())
        selected_bottom.extend(ordered.tail(k)["row"].tolist())

    def mean_cumulative(rows: list[int]) -> np.ndarray:
        if not rows:
            return np.zeros(x_trade.shape[1], dtype=np.float32)
        seq = x_trade[rows, :, 0]
        return np.cumsum(seq, axis=1).mean(axis=0)

    top = mean_cumulative(selected_top)
    bottom = mean_cumulative(selected_bottom)
    return pd.DataFrame({"step": np.arange(1, len(top) + 1), "top_k": top, "bottom_k": bottom})


def plot_outputs(
    output_dir: Path,
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    profile: pd.DataFrame,
    profile_k: int,
) -> None:
    plot_cache = output_dir / "mplconfig"
    plot_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")

    if not daily.empty:
        k0 = int(summary["k"].min())
        pivot = (
            daily[daily["k"] == k0]
            .pivot_table(index="date", columns="method", values="return_after_cost", aggfunc="mean")
            .sort_index()
        )
        cumulative = (1.0 + pivot.fillna(0.0)).cumprod()
        ax = cumulative.plot(figsize=(9, 4), linewidth=1.8)
        ax.set_title(f"Cumulative return after costs, k={k0}")
        ax.set_ylabel("Growth of $1")
        ax.set_xlabel("")
        plt.tight_layout()
        plt.savefig(output_dir / "cumulative_returns_after_cost.png", dpi=180)
        plt.close()

    if not summary.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        for method, group in summary.groupby("method"):
            group = group.sort_values("k")
            ax.plot(group["k"], group["mean_return_before_cost"] * 100.0, marker="o", label=method)
        ax.set_title("Mean daily long-short return before costs")
        ax.set_xlabel("k stocks per leg")
        ax.set_ylabel("Mean daily return (%)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "mean_return_by_k.png", dpi=180)
        plt.close()

    if not profile.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(profile["step"], profile["top_k"], label=f"Top {profile_k}")
        ax.plot(profile["step"], profile["bottom_k"], label=f"Bottom {profile_k}")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title("Average cumulative standardized return sequence")
        ax.set_xlabel("Lookback step")
        ax.set_ylabel("Cumulative standardized return")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "lstm_sequence_profile.png", dpi=180)
        plt.close()


def write_report_draft(
    output_dir: Path,
    args: argparse.Namespace,
    summary: pd.DataFrame,
    period_log: list[dict[str, object]],
) -> None:
    best_k = int(summary["k"].min()) if not summary.empty else 10
    rows = summary[summary["k"] == best_k].sort_values("method")

    result_lines = []
    for row in rows.itertuples(index=False):
        result_lines.append(
            f"| {row.method} | {int(row.k)} | {int(row.days)} | "
            f"{row.mean_return_before_cost * 100:.3f}% | "
            f"{row.mean_return_after_cost * 100:.3f}% | "
            f"{row.sharpe_after_cost:.2f} | {row.accuracy * 100:.2f}% |"
        )

    periods_text = ", ".join(
        f"{p['train_start']} to {p['trade_end']}" for p in period_log
    )
    if args.start or args.end:
        data_range_text = f"{args.start or 'the first available date'} to {args.end or 'the last available date'}"
    else:
        data_range_text = "the available date range"
    result_table = "\n".join(result_lines) if result_lines else "| No valid results | | | | | | |"

    draft = f"""# Replication Draft: Fischer and Krauss (2018)

## Paper Summary

Fischer and Krauss study whether long short-term memory networks can predict the next-day cross-sectional direction of S&P 500 constituent returns. For each rolling study period, they train on 750 trading days and trade the following 250 days. The LSTM input is a sequence of standardized one-day returns, the target is whether the next-day stock return is above the cross-sectional median, and the trading rule goes long the top-ranked stocks and short the bottom-ranked stocks. Their headline k=10 long-short portfolio earns 0.46% mean daily return before costs and 0.26% after a 5 bps half-turn transaction-cost assumption, with a reported annualized Sharpe ratio of 5.83 before costs and 2.34 after costs.

## Replication Design

The supplied `sp500_prices.csv` contains adjusted prices for {data_range_text}. I use simple daily returns, compute the cross-sectional median return each day, standardize returns using training-window mean and standard deviation only, and train one LSTM per rolling period. The implementation keeps the paper's train/trade split and ranking portfolio construction. For runtime, this run used sequence length `{args.seq_len}`, hidden units `{args.hidden}`, max training samples per period `{args.max_train_samples}`, and periods `{periods_text}`. The script can be rerun with `--seq-len 240 --hidden 25 --periods all` for a closer but slower paper-style specification.

## Empirical Results

| Method | k | Days | Mean before costs | Mean after costs | Sharpe after costs | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
{result_table}

The LSTM result should be compared directionally rather than numerically with the paper because the dataset starts in 2000 rather than 1990, extends to 2024 rather than ending in 2015, and uses a reduced training configuration by default. The short-term reversal rule is included because Fischer and Krauss find that the LSTM tends to buy stocks with recent negative return patterns and short stocks with recent positive return patterns.

## Critical Comparison

The original paper finds strong profitability before 2010 and much weaker profitability afterwards. A replication on recent windows is therefore expected to be less spectacular than the 1992-2015 headline number. Similarities would support the paper's interpretation that return-sequence information contains a short-horizon reversal signal. Differences can arise from survivorship/constituent coverage, the available adjusted-price dataset, implementation details, market adaptation after publication, and transaction costs. From a financial analytics perspective, the key insight is not only whether the LSTM beats the benchmark, but whether its selected portfolios show economically interpretable behavior and whether performance survives realistic trading costs.

## Suggested Figures

Use `outputs/cumulative_returns_after_cost.png`, `outputs/mean_return_by_k.png`, and `outputs/lstm_sequence_profile.png` in the report. Keep the final submission under three pages by using one compact results table and one or two figures.
"""
    (output_dir / "report_draft.md").write_text(draft, encoding="utf-8")


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    prices = load_price_panel(args.csv)
    if args.start:
        prices = prices.loc[prices.index >= pd.Timestamp(args.start)]
    if args.end:
        prices = prices.loc[prices.index <= pd.Timestamp(args.end)]
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.dropna(axis=1, how="all")

    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    tickers = returns.columns.tolist()
    raw = returns.to_numpy(dtype=np.float32)
    dates = returns.index
    membership_mask = load_membership_mask(args.membership_csv, dates, tickers)
    medians = compute_signal_medians(raw, membership_mask)

    starts = choose_period_starts(
        len(returns),
        args.train_days,
        args.trade_days,
        args.periods,
        args.period_selection,
    )
    if not starts:
        raise ValueError("Not enough data for the requested train/trade settings")

    settings = LSTMSettings(
        seq_len=args.seq_len,
        hidden_dim=args.hidden,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        dropout=args.dropout,
    )

    all_daily: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []
    all_profiles: list[pd.DataFrame] = []
    history_log: list[dict[str, object]] = []
    period_log: list[dict[str, object]] = []

    for period_no, start in enumerate(starts, start=1):
        train_start = dates[start].date().isoformat()
        train_end = dates[start + args.train_days - 1].date().isoformat()
        trade_start = dates[start + args.train_days].date().isoformat()
        trade_end = dates[start + args.train_days + args.trade_days - 1].date().isoformat()
        print(f"\nPeriod {period_no}/{len(starts)}: train {train_start}..{train_end}, trade {trade_start}..{trade_end}")

        train_slice = raw[start : start + args.train_days]
        mu = float(np.nanmean(train_slice))
        sigma = float(np.nanstd(train_slice))
        if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0.0:
            print("  skipped: invalid training-window standard deviation")
            continue
        std_raw = ((raw - mu) / sigma).astype(np.float32)

        first_train_end = start + args.seq_len - 1
        last_train_end = start + args.train_days - 2
        first_trade_end = start + args.train_days - 1
        last_trade_end = start + args.train_days + args.trade_days - 2

        x_train, y_train, train_meta = build_samples(
            raw,
            std_raw,
            medians,
            dates,
            tickers,
            first_end=first_train_end,
            last_end=last_train_end,
            seq_len=args.seq_len,
            max_samples=args.max_train_samples,
            rng=rng,
            membership_mask=membership_mask,
        )
        x_trade, _, trade_meta = build_samples(
            raw,
            std_raw,
            medians,
            dates,
            tickers,
            first_end=first_trade_end,
            last_end=last_trade_end,
            seq_len=args.seq_len,
            max_samples=None,
            rng=rng,
            membership_mask=membership_mask,
        )
        print(f"  samples: train={len(x_train):,}, trade={len(x_trade):,}, stocks={len(tickers):,}")

        model = NumpyLSTMClassifier(settings, seed=args.seed + period_no)
        history = model.fit(
            x_train,
            y_train,
            validation_order=train_meta["target_idx"].to_numpy(),
        )
        for row in history:
            row = dict(row)
            row["period"] = period_no
            history_log.append(row)

        lstm_scores = model.predict_proba(x_trade)[:, 1]
        reversal_scores = add_reversal_score(trade_meta, raw, horizon=args.reversal_horizon).to_numpy()

        for method, scores in (("LSTM", lstm_scores), (f"Reversal{args.reversal_horizon}D", reversal_scores)):
            daily, summary = evaluate_ranked_portfolios(
                trade_meta,
                scores,
                args.k_values,
                method,
                args.half_turn_cost,
            )
            if not daily.empty:
                daily["period"] = period_no
                all_daily.append(daily)
                all_summary.append(summary)

        profile = selected_sequence_profile(x_trade, trade_meta, lstm_scores, args.profile_k)
        profile["period"] = period_no
        all_profiles.append(profile)
        period_log.append(
            {
                "period": period_no,
                "train_start": train_start,
                "train_end": train_end,
                "trade_start": trade_start,
                "trade_end": trade_end,
                "train_samples": int(len(x_train)),
                "trade_samples": int(len(x_trade)),
            }
        )

    if not all_daily:
        raise RuntimeError("No daily portfolio results were produced")

    daily = pd.concat(all_daily, ignore_index=True)
    summary = (
        daily.groupby(["method", "k"], sort=True)
        .apply(
            lambda group: pd.Series(
                {
                    "days": int(len(group)),
                    "mean_return_before_cost": float(group["return_before_cost"].mean()),
                    "mean_return_after_cost": float(group["return_after_cost"].mean()),
                    "std_after_cost": float(group["return_after_cost"].std(ddof=1)),
                    "sharpe_after_cost": float(
                        group["return_after_cost"].mean()
                        / group["return_after_cost"].std(ddof=1)
                        * math.sqrt(252.0)
                    ),
                    "accuracy": float(group["accuracy"].mean()),
                    "long_mean": float(group["long_return"].mean()),
                    "short_profit_mean": float(group["short_profit"].mean()),
                    "max_drawdown_after_cost": max_drawdown(group.set_index("date")["return_after_cost"]),
                    "positive_days_after_cost": float((group["return_after_cost"] > 0.0).mean()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    profile = (
        pd.concat(all_profiles, ignore_index=True)
        .groupby("step", as_index=False)[["top_k", "bottom_k"]]
        .mean()
    )

    daily.to_csv(output_dir / "daily_portfolio_returns.csv", index=False)
    summary.to_csv(output_dir / "performance_summary.csv", index=False)
    profile.to_csv(output_dir / "lstm_sequence_profile.csv", index=False)
    pd.DataFrame(history_log).to_csv(output_dir / "training_history.csv", index=False)
    (output_dir / "run_config.json").write_text(
        json.dumps({"args": vars(args), "periods": period_log}, indent=2, default=str),
        encoding="utf-8",
    )
    plot_outputs(output_dir, daily, summary, profile, args.profile_k)
    write_report_draft(output_dir, args, summary, period_log)

    print("\nSummary:")
    display_cols = [
        "method",
        "k",
        "days",
        "mean_return_before_cost",
        "mean_return_after_cost",
        "sharpe_after_cost",
        "accuracy",
    ]
    print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:0.5f}"))
    print(f"\nWrote outputs to {output_dir}")
    return daily, summary


def run_subperiods(args: argparse.Namespace) -> None:
    periods = parse_subperiods(args.subperiods)
    if not periods:
        run(args)
        return

    root_output = Path(args.output_dir).expanduser().resolve()
    root_output.mkdir(parents=True, exist_ok=True)
    combined_daily: list[pd.DataFrame] = []
    combined_summary: list[pd.DataFrame] = []

    for label, start, end in periods:
        period_args = copy.deepcopy(args)
        period_args.subperiods = "none"
        period_args.start = start
        period_args.end = end
        period_args.output_dir = str(root_output / label)
        print(f"\n=== LSTM subperiod {label}: {start} to {end} ===")
        daily, summary = run(period_args)
        daily = daily.copy()
        summary = summary.copy()
        daily["subperiod"] = label
        daily["subperiod_start"] = start
        daily["subperiod_end"] = end
        summary["subperiod"] = label
        summary["subperiod_start"] = start
        summary["subperiod_end"] = end
        combined_daily.append(daily)
        combined_summary.append(summary)

    all_daily = pd.concat(combined_daily, ignore_index=True)
    all_summary = pd.concat(combined_summary, ignore_index=True)
    all_daily.to_csv(root_output / "daily_portfolio_returns_by_subperiod.csv", index=False)
    all_summary.to_csv(root_output / "performance_summary_by_subperiod.csv", index=False)

    print("\nCombined LSTM subperiod summary:")
    display_cols = [
        "subperiod",
        "method",
        "k",
        "days",
        "mean_return_before_cost",
        "mean_return_after_cost",
        "sharpe_after_cost",
        "accuracy",
    ]
    print(all_summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:0.5f}"))
    print(f"\nWrote combined LSTM subperiod outputs to {root_output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to sp500_prices.csv")
    parser.add_argument(
        "--membership-csv",
        default=None,
        help="Optional long-form membership snapshot CSV with date,Symbol columns",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for tables and figures")
    parser.add_argument("--start", default=None, help="Optional start date, e.g. 2010-01-01")
    parser.add_argument("--end", default=None, help="Optional end date, e.g. 2024-12-30")
    parser.add_argument("--train-days", type=int, default=750)
    parser.add_argument("--trade-days", type=int, default=250)
    parser.add_argument("--periods", default="all", help="Number of rolling periods to run, or 'all'")
    parser.add_argument("--period-selection", choices=["recent", "early"], default="recent")
    parser.add_argument("--seq-len", type=int, default=240, help="Paper uses 240")
    parser.add_argument("--hidden", type=int, default=25, help="Paper uses 25")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Maximum training samples per rolling period; 0 uses all available samples",
    )
    parser.add_argument("--k-values", nargs="+", default=["10", "50", "100"], type=str)
    parser.add_argument("--profile-k", type=int, default=10)
    parser.add_argument("--reversal-horizon", type=int, default=5)
    parser.add_argument("--half-turn-cost", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--subperiods",
        default="none",
        help="Use 'decades' for 2000-2009, 2010-2019, 2020-2024, or label:start:end entries",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Override core settings for a small smoke/development run",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.quick:
        args.seq_len = 60
        args.hidden = 12
        args.periods = "2"
        args.epochs = 6
        args.max_train_samples = 40000
    args.k_values = parse_k_values(args.k_values)
    run_subperiods(args)


if __name__ == "__main__":
    main()
