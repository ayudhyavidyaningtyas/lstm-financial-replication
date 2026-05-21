#!/usr/bin/env python3
"""Run Fischer-Krauss benchmark models on the S&P 500 panel.

The LSTM paper compares against random forest, a feed-forward DNN, and logistic
regression. These memory-free models use cumulative return features:

    m in {1, ..., 20, 40, 60, ..., 240}

with the same binary target and the same long-top-k/short-bottom-k trading
rule as the LSTM. This script produces a comparison table and figures. It can
also merge in the LSTM daily returns already produced by ``replicate_lstm.py``.
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

from replicate_lstm import (
    DEFAULT_CSV,
    add_reversal_score,
    choose_period_starts,
    evaluate_ranked_portfolios,
    max_drawdown,
    parse_k_values,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs_benchmarks"
DEFAULT_LSTM_DAILY = Path(__file__).resolve().parents[1] / "outputs" / "daily_portfolio_returns.csv"
PAPER_HORIZONS = list(range(1, 21)) + list(range(40, 241, 20))
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


def parse_horizons(value: str) -> list[int]:
    if value.lower() == "paper":
        return PAPER_HORIZONS
    horizons: list[int] = []
    for piece in value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if ":" in piece:
            parts = [int(x) for x in piece.split(":")]
            if len(parts) == 2:
                start, stop = parts
                step = 1
            elif len(parts) == 3:
                start, step, stop = parts
            else:
                raise ValueError(f"Invalid horizon range: {piece}")
            horizons.extend(range(start, stop + 1, step))
        else:
            horizons.append(int(piece))
    return sorted(set(horizons))


def model_list(value: str) -> list[str]:
    allowed = {"logistic", "rf", "dnn", "reversal"}
    models = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(models) - allowed)
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}. Allowed: {', '.join(sorted(allowed))}")
    return models


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


def standardize(train: np.ndarray, trade: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(train, axis=0)
    sigma = np.nanstd(train, axis=0)
    sigma[~np.isfinite(sigma) | (sigma <= 1e-12)] = 1.0
    mu[~np.isfinite(mu)] = 0.0
    return ((train - mu) / sigma).astype(np.float32), ((trade - mu) / sigma).astype(np.float32), mu, sigma


def build_tabular_samples(
    prices: np.ndarray,
    raw_returns: np.ndarray,
    medians: np.ndarray,
    dates: pd.DatetimeIndex,
    tickers: list[str],
    *,
    first_end: int,
    last_end: int,
    horizons: list[int],
    max_samples: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if last_end < first_end:
        raise ValueError("No tabular endpoints available")

    end_indices = np.arange(first_end, last_end + 1)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    meta_parts: list[pd.DataFrame] = []

    for stock_idx, ticker in enumerate(tickers):
        now = prices[end_indices, stock_idx]
        feature_columns = []
        for horizon in horizons:
            lagged = prices[end_indices - horizon, stock_idx]
            feature_columns.append(now / lagged - 1.0)
        features = np.column_stack(feature_columns)

        target_indices = end_indices + 1
        next_returns = raw_returns[target_indices, stock_idx]
        target_medians = medians[target_indices]
        valid = np.isfinite(features).all(axis=1) & np.isfinite(next_returns) & np.isfinite(target_medians)
        if not np.any(valid):
            continue

        labels = (next_returns[valid] >= target_medians[valid]).astype(np.int64)
        x_parts.append(features[valid].astype(np.float32, copy=True))
        y_parts.append(labels)
        meta_parts.append(
            pd.DataFrame(
                {
                    "target_date": dates[target_indices[valid]],
                    "end_idx": end_indices[valid],
                    "target_idx": target_indices[valid],
                    "ticker": ticker,
                    "stock_idx": stock_idx,
                    "realized_return": next_returns[valid].astype(np.float32),
                    "label": labels,
                }
            )
        )

    if not x_parts:
        raise ValueError("No valid tabular samples after filtering missing values")

    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    meta = pd.concat(meta_parts, ignore_index=True)

    if max_samples is not None and len(x) > max_samples:
        keep = rng.choice(len(x), size=max_samples, replace=False)
        x = x[keep]
        y = y[keep]
        meta = meta.iloc[keep].reset_index(drop=True)

    return x, y, meta.reset_index(drop=True)


class LogisticRegressionSGD:
    def __init__(
        self,
        *,
        learning_rate: float,
        l2: float,
        epochs: int,
        batch_size: int,
        patience: int,
        seed: int,
    ) -> None:
        self.learning_rate = learning_rate
        self.l2 = l2
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.rng = np.random.default_rng(seed)
        self.w: np.ndarray | None = None
        self.b = 0.0
        self.history: list[dict[str, float]] = []

    def _predict_raw(self, x: np.ndarray) -> np.ndarray:
        assert self.w is not None
        return sigmoid(x @ self.w + self.b)

    def _loss(self, x: np.ndarray, y: np.ndarray) -> float:
        p = self._predict_raw(x)
        loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
        assert self.w is not None
        return float(loss + 0.5 * self.l2 * np.sum(self.w * self.w))

    def fit(self, x: np.ndarray, y: np.ndarray) -> list[dict[str, float]]:
        n, p = x.shape
        self.w = self.rng.normal(0.0, 0.01, size=p).astype(np.float32)
        self.b = 0.0
        cache_w = np.zeros_like(self.w)
        cache_b = 0.0

        order = self.rng.permutation(n)
        split = int(n * 0.8)
        train_idx, val_idx = order[:split], order[split:]
        x_train, y_train = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]

        best_loss = math.inf
        best_w = self.w.copy()
        best_b = self.b
        bad_epochs = 0
        self.history = []

        for epoch in range(1, self.epochs + 1):
            epoch_order = self.rng.permutation(len(x_train))
            for start in range(0, len(epoch_order), self.batch_size):
                idx = epoch_order[start : start + self.batch_size]
                xb, yb = x_train[idx], y_train[idx]
                pred = self._predict_raw(xb)
                error = pred - yb
                grad_w = xb.T @ error / len(xb) + self.l2 * self.w
                grad_b = float(np.mean(error))
                cache_w = 0.9 * cache_w + 0.1 * grad_w * grad_w
                cache_b = 0.9 * cache_b + 0.1 * grad_b * grad_b
                self.w -= self.learning_rate * grad_w / (np.sqrt(cache_w) + 1e-7)
                self.b -= self.learning_rate * grad_b / (math.sqrt(cache_b) + 1e-7)

            train_loss = self._loss(x_train[: min(len(x_train), 20000)], y_train[: min(len(y_train), 20000)])
            val_loss = self._loss(x_val, y_val) if len(x_val) else train_loss
            self.history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
            print(f"    LOG epoch {epoch:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
            if val_loss + 1e-5 < best_loss:
                best_loss = val_loss
                best_w = self.w.copy()
                best_b = self.b
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        self.w = best_w
        self.b = best_b
        return self.history

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p1 = self._predict_raw(x)
        return np.column_stack([1.0 - p1, p1])


class DenseNeuralNetwork:
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_layers: list[int],
        learning_rate: float,
        l2: float,
        dropout: float,
        epochs: int,
        batch_size: int,
        patience: int,
        seed: int,
    ) -> None:
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.l2 = l2
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.rng = np.random.default_rng(seed)
        dims = [input_dim] + hidden_layers + [2]
        self.params: dict[str, np.ndarray] = {}
        for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:]), start=1):
            scale = math.sqrt(2.0 / max(din, 1))
            self.params[f"W{i}"] = self.rng.normal(0.0, scale, size=(din, dout)).astype(np.float32)
            self.params[f"b{i}"] = np.zeros(dout, dtype=np.float32)
        self.cache = {name: np.zeros_like(value) for name, value in self.params.items()}
        self.history: list[dict[str, float]] = []

    @property
    def n_layers(self) -> int:
        return len(self.hidden_layers) + 1

    def _forward(
        self,
        x: np.ndarray,
        y: np.ndarray | None = None,
        *,
        training: bool,
        store_cache: bool,
    ) -> tuple[float | None, np.ndarray, dict[str, np.ndarray] | None]:
        activations: dict[str, np.ndarray] | None = {} if store_cache else None
        a = x
        if activations is not None:
            activations["a0"] = a

        for i in range(1, self.n_layers):
            z = a @ self.params[f"W{i}"] + self.params[f"b{i}"]
            a = np.maximum(z, 0.0)
            if training and self.dropout > 0.0:
                keep = 1.0 - self.dropout
                mask = (self.rng.random(a.shape) < keep).astype(np.float32) / keep
                a = a * mask
                if activations is not None:
                    activations[f"mask{i}"] = mask
            if activations is not None:
                activations[f"z{i}"] = z
                activations[f"a{i}"] = a

        logits = a @ self.params[f"W{self.n_layers}"] + self.params[f"b{self.n_layers}"]
        probs = softmax(logits)
        loss = None
        if y is not None:
            loss = float(-np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12)))
            loss += float(0.5 * self.l2 * sum(np.sum(v * v) for k, v in self.params.items() if k.startswith("W")))
        return loss, probs, activations

    def _backward(
        self,
        probs: np.ndarray,
        y: np.ndarray,
        cache: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        grads = {name: np.zeros_like(value) for name, value in self.params.items()}
        delta = probs.copy()
        delta[np.arange(len(y)), y] -= 1.0
        delta /= len(y)

        for i in reversed(range(1, self.n_layers + 1)):
            a_prev = cache[f"a{i - 1}"]
            grads[f"W{i}"] = a_prev.T @ delta + self.l2 * self.params[f"W{i}"]
            grads[f"b{i}"] = delta.sum(axis=0)
            if i > 1:
                delta = delta @ self.params[f"W{i}"].T
                if f"mask{i - 1}" in cache:
                    delta = delta * cache[f"mask{i - 1}"]
                delta = delta * (cache[f"z{i - 1}"] > 0.0)

        norm = math.sqrt(float(sum(np.sum(g * g) for g in grads.values())))
        if norm > 5.0:
            scale = 5.0 / (norm + 1e-12)
            grads = {name: grad * scale for name, grad in grads.items()}
        return grads

    def _apply(self, grads: dict[str, np.ndarray]) -> None:
        for name, grad in grads.items():
            self.cache[name] = 0.9 * self.cache[name] + 0.1 * grad * grad
            self.params[name] -= self.learning_rate * grad / (np.sqrt(self.cache[name]) + 1e-7)

    def _loss_in_batches(self, x: np.ndarray, y: np.ndarray) -> float:
        total = 0.0
        n = 0
        for start in range(0, len(x), self.batch_size):
            xb = x[start : start + self.batch_size]
            yb = y[start : start + self.batch_size]
            loss, _, _ = self._forward(xb, yb, training=False, store_cache=False)
            if loss is not None:
                total += loss * len(xb)
                n += len(xb)
        return total / max(n, 1)

    def fit(self, x: np.ndarray, y: np.ndarray) -> list[dict[str, float]]:
        order = self.rng.permutation(len(x))
        split = int(len(x) * 0.8)
        train_idx, val_idx = order[:split], order[split:]
        x_train, y_train = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]

        best_loss = math.inf
        best_params = {name: value.copy() for name, value in self.params.items()}
        bad_epochs = 0
        self.history = []

        for epoch in range(1, self.epochs + 1):
            epoch_order = self.rng.permutation(len(x_train))
            train_total = 0.0
            train_n = 0
            for start in range(0, len(epoch_order), self.batch_size):
                idx = epoch_order[start : start + self.batch_size]
                xb, yb = x_train[idx], y_train[idx]
                loss, probs, cache = self._forward(xb, yb, training=True, store_cache=True)
                assert cache is not None and loss is not None
                grads = self._backward(probs, yb, cache)
                self._apply(grads)
                train_total += loss * len(xb)
                train_n += len(xb)

            train_loss = train_total / max(train_n, 1)
            val_loss = self._loss_in_batches(x_val, y_val) if len(x_val) else train_loss
            self.history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
            print(f"    DNN epoch {epoch:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
            if val_loss + 1e-5 < best_loss:
                best_loss = val_loss
                best_params = {name: value.copy() for name, value in self.params.items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        self.params = best_params
        return self.history

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        chunks = []
        for start in range(0, len(x), self.batch_size):
            _, probs, _ = self._forward(x[start : start + self.batch_size], training=False, store_cache=False)
            chunks.append(probs)
        return np.vstack(chunks)


@dataclass
class TreeNode:
    proba: float
    feature: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


class SimpleDecisionTree:
    def __init__(
        self,
        *,
        max_depth: int,
        max_features: int,
        min_samples_leaf: int,
        n_thresholds: int,
        seed: int,
    ) -> None:
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.n_thresholds = n_thresholds
        self.rng = np.random.default_rng(seed)
        self.root: TreeNode | None = None

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        p = float(np.mean(y))
        return 2.0 * p * (1.0 - p)

    def fit(self, x: np.ndarray, y: np.ndarray, indices: np.ndarray) -> None:
        self.root = self._build(x, y, indices, depth=0)

    def _best_split(
        self,
        x: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
    ) -> tuple[int | None, float | None, np.ndarray | None]:
        n_features = x.shape[1]
        candidate_features = self.rng.choice(
            n_features,
            size=min(self.max_features, n_features),
            replace=False,
        )
        y_node = y[indices]
        parent_impurity = self._gini(y_node)
        best_gain = 0.0
        best_feature: int | None = None
        best_threshold: float | None = None
        best_left_mask: np.ndarray | None = None

        for feature in candidate_features:
            values = x[indices, feature]
            if np.nanmin(values) == np.nanmax(values):
                continue
            quantiles = np.linspace(0.0, 1.0, self.n_thresholds + 2)[1:-1]
            thresholds = np.unique(np.quantile(values, quantiles))
            for threshold in thresholds:
                left_mask = values <= threshold
                left_n = int(left_mask.sum())
                right_n = len(indices) - left_n
                if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                    continue
                left_y = y_node[left_mask]
                right_y = y_node[~left_mask]
                impurity = (left_n / len(indices)) * self._gini(left_y)
                impurity += (right_n / len(indices)) * self._gini(right_y)
                gain = parent_impurity - impurity
                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(feature)
                    best_threshold = float(threshold)
                    best_left_mask = left_mask.copy()

        return best_feature, best_threshold, best_left_mask

    def _build(self, x: np.ndarray, y: np.ndarray, indices: np.ndarray, depth: int) -> TreeNode:
        y_node = y[indices]
        proba = float(np.mean(y_node)) if len(y_node) else 0.5
        if (
            depth >= self.max_depth
            or len(indices) < 2 * self.min_samples_leaf
            or proba <= 1e-6
            or proba >= 1.0 - 1e-6
        ):
            return TreeNode(proba=proba)

        feature, threshold, left_mask = self._best_split(x, y, indices)
        if feature is None or threshold is None or left_mask is None:
            return TreeNode(proba=proba)

        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]
        return TreeNode(
            proba=proba,
            feature=feature,
            threshold=threshold,
            left=self._build(x, y, left_indices, depth + 1),
            right=self._build(x, y, right_indices, depth + 1),
        )

    def predict_proba_1d(self, x: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Tree is not fitted")
        out = np.empty(len(x), dtype=np.float32)
        stack: list[tuple[TreeNode, np.ndarray]] = [(self.root, np.arange(len(x)))]
        while stack:
            node, indices = stack.pop()
            if node.feature is None or node.threshold is None or node.left is None or node.right is None:
                out[indices] = node.proba
                continue
            left_mask = x[indices, node.feature] <= node.threshold
            if np.any(left_mask):
                stack.append((node.left, indices[left_mask]))
            if np.any(~left_mask):
                stack.append((node.right, indices[~left_mask]))
        return out


class SimpleRandomForest:
    def __init__(
        self,
        *,
        n_estimators: int,
        max_depth: int,
        min_samples_leaf: int,
        n_thresholds: int,
        seed: int,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_thresholds = n_thresholds
        self.rng = np.random.default_rng(seed)
        self.trees: list[SimpleDecisionTree] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> list[dict[str, float]]:
        max_features = max(1, int(math.sqrt(x.shape[1])))
        self.trees = []
        for i in range(1, self.n_estimators + 1):
            indices = self.rng.integers(0, len(x), size=len(x))
            tree = SimpleDecisionTree(
                max_depth=self.max_depth,
                max_features=max_features,
                min_samples_leaf=self.min_samples_leaf,
                n_thresholds=self.n_thresholds,
                seed=int(self.rng.integers(0, 2**31 - 1)),
            )
            tree.fit(x, y, indices)
            self.trees.append(tree)
            if i == 1 or i % max(1, self.n_estimators // 5) == 0:
                print(f"    RF tree {i}/{self.n_estimators}")
        return [{"epoch": float(len(self.trees)), "train_loss": np.nan, "val_loss": np.nan}]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p1 = np.zeros(len(x), dtype=np.float64)
        for tree in self.trees:
            p1 += tree.predict_proba_1d(x)
        p1 /= max(len(self.trees), 1)
        return np.column_stack([1.0 - p1, p1])


def fit_random_forest(
    x: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[object, list[dict[str, float]], str]:
    if args.use_sklearn_rf:
        try:
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(
                n_estimators=args.rf_trees,
                max_depth=args.rf_depth,
                min_samples_leaf=args.rf_min_samples_leaf,
                max_features="sqrt",
                n_jobs=-1,
                random_state=seed,
                bootstrap=True,
            )
            model.fit(x, y)
            return model, [{"epoch": float(args.rf_trees), "train_loss": np.nan, "val_loss": np.nan}], "sklearn"
        except Exception as exc:
            print(f"    sklearn RF unavailable ({exc}); using NumPy fallback")

    model = SimpleRandomForest(
        n_estimators=args.rf_trees,
        max_depth=args.rf_depth,
        min_samples_leaf=args.rf_min_samples_leaf,
        n_thresholds=args.rf_thresholds,
        seed=seed,
    )
    history = model.fit(x, y)
    return model, history, "numpy"


def summarise_daily(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, k), group in daily.groupby(["method", "k"], sort=True):
        before = group["return_before_cost"]
        after = group["return_after_cost"]
        std_after = float(after.std(ddof=1))
        rows.append(
            {
                "method": method,
                "k": int(k),
                "days": int(len(group)),
                "mean_return_before_cost": float(before.mean()),
                "mean_return_after_cost": float(after.mean()),
                "std_after_cost": std_after,
                "sharpe_after_cost": float(after.mean() / std_after * math.sqrt(252.0)) if std_after > 0 else np.nan,
                "accuracy": float(group["accuracy"].mean()),
                "long_mean": float(group["long_return"].mean()),
                "short_profit_mean": float(group["short_profit"].mean()),
                "max_drawdown_after_cost": max_drawdown(group.set_index("date")["return_after_cost"]),
                "positive_days_after_cost": float((after > 0.0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["k", "method"]).reset_index(drop=True)


def plot_benchmark_outputs(output_dir: Path, daily: pd.DataFrame, summary: pd.DataFrame) -> None:
    plot_cache = output_dir / "mplconfig"
    plot_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    k0 = int(summary["k"].min())
    pivot = (
        daily[daily["k"] == k0]
        .pivot_table(index="date", columns="method", values="return_after_cost", aggfunc="mean")
        .sort_index()
    )
    cumulative = (1.0 + pivot.fillna(0.0)).cumprod()
    ax = cumulative.plot(figsize=(9, 4), linewidth=1.6)
    ax.set_title(f"Model comparison cumulative return after costs, k={k0}")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "benchmark_cumulative_after_cost.png", dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    for method, group in summary.groupby("method"):
        group = group.sort_values("k")
        ax.plot(group["k"], group["mean_return_before_cost"] * 100.0, marker="o", label=method)
    ax.set_title("Benchmark mean daily return before costs")
    ax.set_xlabel("k stocks per leg")
    ax.set_ylabel("Mean daily return (%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "benchmark_mean_return_by_k.png", dpi=180)
    plt.close()


def write_report_snippet(output_dir: Path, summary: pd.DataFrame, args: argparse.Namespace) -> None:
    k0 = int(summary["k"].min())
    rows = summary[summary["k"] == k0].sort_values("mean_return_before_cost", ascending=False)
    table_rows = [
        "| Method | k | Days | Mean before costs | Mean after costs | Sharpe after costs | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows.itertuples(index=False):
        table_rows.append(
            f"| {row.method} | {int(row.k)} | {int(row.days)} | "
            f"{row.mean_return_before_cost * 100:.3f}% | "
            f"{row.mean_return_after_cost * 100:.3f}% | "
            f"{row.sharpe_after_cost:.2f} | {row.accuracy * 100:.2f}% |"
        )

    text = f"""# Benchmark Comparison Snippet

The benchmark models use the paper's memory-free feature set: cumulative returns over horizons `{args.horizons}`. They are evaluated with the same cross-sectional median target and the same long-top-k/short-bottom-k trading rule as the LSTM.

{chr(10).join(table_rows)}

Interpretation guide:

- Logistic regression tests whether a linear model can extract the signal.
- Random forest tests nonlinear interactions without sequence memory.
- DNN tests a feed-forward neural network without recurrent memory.
- Reversal5D tests whether a simple short-term reversal rule explains most of the performance.
- If LSTM is included, it is read from the existing `replicate_lstm.py` output rather than retrained here.
"""
    (output_dir / "benchmark_report_snippet.md").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    models = model_list(args.models)
    horizons = parse_horizons(args.horizons)
    max_horizon = max(horizons)

    prices_df = pd.read_csv(args.csv, parse_dates=["Date"]).set_index("Date").sort_index()
    if args.start:
        prices_df = prices_df.loc[prices_df.index >= pd.Timestamp(args.start)]
    if args.end:
        prices_df = prices_df.loc[prices_df.index <= pd.Timestamp(args.end)]
    prices_df = prices_df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

    returns_df = prices_df.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    prices = prices_df.to_numpy(dtype=np.float32)
    raw = returns_df.to_numpy(dtype=np.float32)
    dates = returns_df.index
    tickers = returns_df.columns.tolist()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        medians = np.nanmedian(raw, axis=1)

    starts = choose_period_starts(
        len(returns_df),
        args.train_days,
        args.trade_days,
        args.periods,
        args.period_selection,
    )
    if not starts:
        raise ValueError("Not enough data for requested benchmark windows")

    all_daily: list[pd.DataFrame] = []
    history_rows: list[dict[str, object]] = []
    period_rows: list[dict[str, object]] = []

    for period_no, start in enumerate(starts, start=1):
        train_start = dates[start].date().isoformat()
        train_end = dates[start + args.train_days - 1].date().isoformat()
        trade_start = dates[start + args.train_days].date().isoformat()
        trade_end = dates[start + args.train_days + args.trade_days - 1].date().isoformat()
        print(f"\nPeriod {period_no}/{len(starts)}: train {train_start}..{train_end}, trade {trade_start}..{trade_end}")

        first_train_end = start + max_horizon
        last_train_end = start + args.train_days - 2
        first_trade_end = start + args.train_days - 1
        last_trade_end = start + args.train_days + args.trade_days - 2

        x_train_raw, y_train, _ = build_tabular_samples(
            prices,
            raw,
            medians,
            dates,
            tickers,
            first_end=first_train_end,
            last_end=last_train_end,
            horizons=horizons,
            max_samples=args.max_train_samples,
            rng=rng,
        )
        x_trade_raw, _, trade_meta = build_tabular_samples(
            prices,
            raw,
            medians,
            dates,
            tickers,
            first_end=first_trade_end,
            last_end=last_trade_end,
            horizons=horizons,
            max_samples=None,
            rng=rng,
        )
        x_train, x_trade, _, _ = standardize(x_train_raw, x_trade_raw)
        print(f"  samples: train={len(x_train):,}, trade={len(x_trade):,}, features={x_train.shape[1]}")

        if "logistic" in models:
            log_model = LogisticRegressionSGD(
                learning_rate=args.log_lr,
                l2=args.log_l2,
                epochs=args.log_epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                seed=args.seed + 1000 + period_no,
            )
            history = log_model.fit(x_train, y_train)
            scores = log_model.predict_proba(x_trade)[:, 1]
            daily, _ = evaluate_ranked_portfolios(trade_meta, scores, args.k_values, "LOG", args.half_turn_cost)
            daily["period"] = period_no
            all_daily.append(daily)
            for row in history:
                history_rows.append({"period": period_no, "method": "LOG", **row})

        if "dnn" in models:
            dnn_model = DenseNeuralNetwork(
                input_dim=x_train.shape[1],
                hidden_layers=args.dnn_hidden,
                learning_rate=args.dnn_lr,
                l2=args.dnn_l2,
                dropout=args.dnn_dropout,
                epochs=args.dnn_epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                seed=args.seed + 2000 + period_no,
            )
            history = dnn_model.fit(x_train, y_train)
            scores = dnn_model.predict_proba(x_trade)[:, 1]
            daily, _ = evaluate_ranked_portfolios(trade_meta, scores, args.k_values, "DNN", args.half_turn_cost)
            daily["period"] = period_no
            all_daily.append(daily)
            for row in history:
                history_rows.append({"period": period_no, "method": "DNN", **row})

        if "rf" in models:
            rf_model, history, backend = fit_random_forest(
                x_train,
                y_train,
                args,
                seed=args.seed + 3000 + period_no,
            )
            scores = rf_model.predict_proba(x_trade)[:, 1]
            daily, _ = evaluate_ranked_portfolios(trade_meta, scores, args.k_values, "RAF", args.half_turn_cost)
            daily["period"] = period_no
            daily["rf_backend"] = backend
            all_daily.append(daily)
            for row in history:
                history_rows.append({"period": period_no, "method": "RAF", "backend": backend, **row})

        if "reversal" in models:
            reversal_scores = add_reversal_score(trade_meta, raw, horizon=args.reversal_horizon).to_numpy()
            daily, _ = evaluate_ranked_portfolios(
                trade_meta,
                reversal_scores,
                args.k_values,
                f"Reversal{args.reversal_horizon}D",
                args.half_turn_cost,
            )
            daily["period"] = period_no
            all_daily.append(daily)

        period_rows.append(
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

    if args.include_lstm and args.lstm_daily and Path(args.lstm_daily).exists():
        lstm_daily = pd.read_csv(args.lstm_daily, parse_dates=["date"])
        lstm_daily = lstm_daily[lstm_daily["method"].eq("LSTM") & lstm_daily["k"].isin(args.k_values)].copy()
        if args.start:
            lstm_daily = lstm_daily[lstm_daily["date"] >= pd.Timestamp(args.start)]
        if args.end:
            lstm_daily = lstm_daily[lstm_daily["date"] <= pd.Timestamp(args.end)]
        if not lstm_daily.empty:
            print(f"\nIncluding existing LSTM daily returns from {args.lstm_daily}")
            all_daily.append(lstm_daily)

    if not all_daily:
        raise RuntimeError("No benchmark daily returns produced")

    daily = pd.concat(all_daily, ignore_index=True, sort=False)
    summary = summarise_daily(daily)

    daily.to_csv(output_dir / "benchmark_daily_returns.csv", index=False)
    summary.to_csv(output_dir / "benchmark_performance_summary.csv", index=False)
    pd.DataFrame(history_rows).to_csv(output_dir / "benchmark_training_history.csv", index=False)
    (output_dir / "benchmark_run_config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "horizons": horizons,
                "periods": period_rows,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    plot_benchmark_outputs(output_dir, daily, summary)
    write_report_snippet(output_dir, summary, args)

    display_cols = [
        "method",
        "k",
        "days",
        "mean_return_before_cost",
        "mean_return_after_cost",
        "sharpe_after_cost",
        "accuracy",
    ]
    print("\nBenchmark summary:")
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
        print(f"\n=== Benchmark subperiod {label}: {start} to {end} ===")
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
    all_daily.to_csv(root_output / "benchmark_daily_returns_by_subperiod.csv", index=False)
    all_summary.to_csv(root_output / "benchmark_performance_summary_by_subperiod.csv", index=False)

    print("\nCombined benchmark subperiod summary:")
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
    print(f"\nWrote combined benchmark subperiod outputs to {root_output}")


def parse_hidden_layers(value: str) -> list[int]:
    return [int(piece.strip()) for piece in value.split(",") if piece.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--train-days", type=int, default=750)
    parser.add_argument("--trade-days", type=int, default=250)
    parser.add_argument("--periods", default="all", help="Number of rolling periods, or 'all'")
    parser.add_argument("--period-selection", choices=["recent", "early"], default="recent")
    parser.add_argument("--horizons", default="paper", help="'paper' or comma/range list, e.g. 1:20,40:20:240")
    parser.add_argument("--models", default="logistic,rf,dnn,reversal")
    parser.add_argument("--k-values", nargs="+", default=["10", "50", "100"], type=str)
    parser.add_argument("--max-train-samples", type=int, default=120000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--half-turn-cost", type=float, default=0.0005)
    parser.add_argument("--reversal-horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--log-epochs", type=int, default=12)
    parser.add_argument("--log-lr", type=float, default=0.01)
    parser.add_argument("--log-l2", type=float, default=1e-4)

    parser.add_argument("--dnn-hidden", type=parse_hidden_layers, default=parse_hidden_layers("31,10,5"))
    parser.add_argument("--dnn-epochs", type=int, default=12)
    parser.add_argument("--dnn-lr", type=float, default=0.001)
    parser.add_argument("--dnn-l2", type=float, default=1e-5)
    parser.add_argument("--dnn-dropout", type=float, default=0.2)

    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--rf-depth", type=int, default=12)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=100)
    parser.add_argument("--rf-thresholds", type=int, default=12)
    parser.add_argument("--no-sklearn-rf", action="store_false", dest="use_sklearn_rf")
    parser.set_defaults(use_sklearn_rf=True)

    parser.add_argument("--lstm-daily", default=str(DEFAULT_LSTM_DAILY))
    parser.add_argument("--no-include-lstm", action="store_false", dest="include_lstm")
    parser.set_defaults(include_lstm=True)
    parser.add_argument(
        "--subperiods",
        default="none",
        help="Use 'decades' for 2000-2009, 2010-2019, 2020-2024, or label:start:end entries",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.k_values = parse_k_values(args.k_values)
    run_subperiods(args)


if __name__ == "__main__":
    main()
