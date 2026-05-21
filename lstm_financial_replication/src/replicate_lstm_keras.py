#!/usr/bin/env python3
"""Keras/TensorFlow LSTM runner for the Fischer and Krauss replication.

This script intentionally reuses the data preparation, rolling windows,
membership masking, portfolio evaluation, and plotting helpers from
``replicate_lstm.py``. The only experimental change is the LSTM implementation:
TensorFlow/Keras replaces the compact NumPy training loop.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from replicate_lstm import (
    DEFAULT_CSV,
    PROJECT_ROOT,
    add_reversal_score,
    build_samples,
    choose_period_starts,
    chronological_train_val_indices,
    compute_signal_medians,
    evaluate_ranked_portfolios,
    load_membership_mask,
    load_price_panel,
    max_drawdown,
    parse_k_values,
    parse_subperiods,
    plot_outputs,
    selected_sequence_profile,
    write_report_draft,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs_lstm_keras"


def import_tensorflow():
    """Import TensorFlow lazily so the error message can be coursework-friendly."""
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "TensorFlow is not installed in this Python environment.\n\n"
            "Create a compatible environment, then rerun the command:\n\n"
            "  conda create -n lstm-keras python=3.11 -y\n"
            "  conda activate lstm-keras\n"
            "  pip install -r requirements.txt\n"
            "  pip install -r requirements-tensorflow.txt\n\n"
            "Then run this script from inside lstm_financial_replication/."
        ) from exc
    return tf


def set_tensorflow_reproducibility(tf, seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_keras_lstm(args: argparse.Namespace, tf):
    regularizer = None
    if args.l2 > 0.0:
        regularizer = tf.keras.regularizers.l2(args.l2)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(args.seq_len, 1)),
            tf.keras.layers.LSTM(
                args.hidden,
                dropout=args.dropout,
                recurrent_dropout=args.recurrent_dropout,
                kernel_regularizer=regularizer,
                recurrent_regularizer=regularizer,
            ),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )

    optimizer_name = args.optimizer.lower()
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate,
            clipnorm=args.gradient_clip if args.gradient_clip > 0 else None,
        )
    elif optimizer_name == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=args.learning_rate,
            rho=args.rmsprop_rho,
            clipnorm=args.gradient_clip if args.gradient_clip > 0 else None,
        )
    else:
        raise ValueError("--optimizer must be 'rmsprop' or 'adam'")

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fit_keras_model(
    model,
    x: np.ndarray,
    y: np.ndarray,
    target_order: np.ndarray,
    args: argparse.Namespace,
    tf,
) -> pd.DataFrame:
    train_idx, val_idx = chronological_train_val_indices(
        len(x),
        args.validation_fraction,
        target_order,
    )
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            min_delta=args.min_delta,
            restore_best_weights=True,
        )
    ]
    if args.reduce_lr_patience > 0:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=args.reduce_lr_patience,
                min_delta=args.min_delta,
                min_lr=args.min_learning_rate,
                verbose=0,
            )
        )

    class PrintEpoch(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            print(
                f"    epoch {epoch + 1:02d}: "
                f"train_loss={logs.get('loss', float('nan')):.4f} "
                f"val_loss={logs.get('val_loss', float('nan')):.4f} "
                f"val_acc={logs.get('val_accuracy', float('nan')):.4f}"
            )

    callbacks.append(PrintEpoch())

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        verbose=0,
        callbacks=callbacks,
    )

    rows = []
    for i, loss in enumerate(history.history.get("loss", []), start=1):
        rows.append(
            {
                "epoch": i,
                "train_loss": float(loss),
                "val_loss": float(history.history.get("val_loss", [np.nan] * i)[i - 1]),
                "train_accuracy": float(history.history.get("accuracy", [np.nan] * i)[i - 1]),
                "val_accuracy": float(history.history.get("val_accuracy", [np.nan] * i)[i - 1]),
                "learning_rate": float(history.history.get("learning_rate", [np.nan] * i)[i - 1]),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace, tf) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    all_daily: list[pd.DataFrame] = []
    all_profiles: list[pd.DataFrame] = []
    history_log: list[pd.DataFrame] = []
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

        tf.keras.backend.clear_session()
        set_tensorflow_reproducibility(tf, args.seed + period_no)
        model = build_keras_lstm(args, tf)
        history = fit_keras_model(
            model,
            x_train,
            y_train,
            train_meta["target_idx"].to_numpy(),
            args,
            tf,
        )
        if not history.empty:
            history.insert(0, "period", period_no)
            history_log.append(history)

        lstm_scores = model.predict(x_trade, batch_size=args.predict_batch_size, verbose=0)[:, 1]
        reversal_scores = add_reversal_score(trade_meta, raw, horizon=args.reversal_horizon).to_numpy()

        for method, scores in (("KerasLSTM", lstm_scores), (f"Reversal{args.reversal_horizon}D", reversal_scores)):
            daily, _ = evaluate_ranked_portfolios(
                trade_meta,
                scores,
                args.k_values,
                method,
                args.half_turn_cost,
            )
            if not daily.empty:
                daily["period"] = period_no
                all_daily.append(daily)

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
    if history_log:
        pd.concat(history_log, ignore_index=True).to_csv(output_dir / "training_history.csv", index=False)
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


def run_subperiods(args: argparse.Namespace, tf) -> None:
    periods = parse_subperiods(args.subperiods)
    if not periods:
        run(args, tf)
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
        print(f"\n=== Keras LSTM subperiod {label}: {start} to {end} ===")
        daily, summary = run(period_args, tf)
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

    print("\nCombined Keras LSTM subperiod summary:")
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
    print(f"\nWrote combined Keras LSTM subperiod outputs to {root_output}")


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
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--predict-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--optimizer", choices=["rmsprop", "adam"], default="rmsprop")
    parser.add_argument("--rmsprop-rho", type=float, default=0.9)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--recurrent-dropout", type=float, default=0.0)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--reduce-lr-patience", type=int, default=3)
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
    args = build_parser().parse_args()
    if args.quick:
        args.seq_len = 60
        args.hidden = 12
        args.periods = "1"
        args.epochs = 3
        args.max_train_samples = 20000
        args.batch_size = 256
    args.k_values = parse_k_values(args.k_values)
    tf = import_tensorflow()
    set_tensorflow_reproducibility(tf, args.seed)
    run_subperiods(args, tf)


if __name__ == "__main__":
    main()
