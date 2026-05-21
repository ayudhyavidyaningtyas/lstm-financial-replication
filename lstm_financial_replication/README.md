# LSTM Financial Market Replication

This folder contains a coursework-oriented replication of Fischer and Krauss
(2018), "Deep learning with long short-term memory networks for financial market
predictions". Generated price files live in `data/` and are not committed.

## What It Replicates

- Rolling study periods: 750 trading days for training and 250 for out-of-sample trading.
- Inputs: sequences of standardized one-day stock returns.
- Target: whether next-day stock return beats the cross-sectional median.
- Portfolio rule: rank stocks by predicted probability, go long the top `k`, short the bottom `k`.
- Costs: 5 bps per half-turn, matching the assumption used in the paper.
- Benchmark: a transparent 5-day short-term reversal strategy inspired by the paper's black-box analysis.
- Validation: the LSTM, DNN, and logistic models hold out the most recent dates inside each training window for validation.

The implementation uses only `numpy`, `pandas`, and `matplotlib`, because the local environment does not currently include TensorFlow or PyTorch. The LSTM is a compact NumPy implementation with RMSprop and early stopping.

## Setup Data

```bash
python3 src/download_sp500_prices.py \
  --start 2000-01-01 \
  --end 2024-12-31 \
  --output-dir data
```

This creates the default input file:

```text
data/sp500_prices.csv
```

## Headline LSTM Run

The default LSTM settings are paper-style: 240-day sequences, 25 hidden units,
750-day training windows, 250-day trading windows, and all available rolling
periods. Early stopping now uses chronological validation dates, with 30 maximum
epochs and patience 5. The default sample cap is disabled, so each rolling
period uses all available training samples.

```bash
python3 src/replicate_lstm.py
```

For a fast smoke/development run:

```bash
python3 src/replicate_lstm.py --quick
```

## Subperiod Runs

To run the LSTM once across the three report subperiods:

```bash
python3 src/replicate_lstm.py \
  --csv data/sp500_prices.csv \
  --subperiods decades \
  --seq-len 240 \
  --hidden 25 \
  --periods all \
  --epochs 30 \
  --patience 5 \
  --max-train-samples 0 \
  --output-dir outputs_lstm_subperiods
```

This writes one folder per subperiod plus:

- `outputs_lstm_subperiods/performance_summary_by_subperiod.csv`
- `outputs_lstm_subperiods/daily_portfolio_returns_by_subperiod.csv`

The preset subperiods are:

- `2000_2009`: 2000-01-01 to 2009-12-31
- `2010_2019`: 2010-01-01 to 2019-12-31
- `2020_2024`: 2020-01-01 to 2024-12-31

## Outputs

The script writes these files to `lstm_financial_replication/outputs/`:

- `performance_summary.csv`: main results table by method and portfolio size.
- `daily_portfolio_returns.csv`: daily long-short returns for plotting and audit.
- `training_history.csv`: LSTM train/validation losses by rolling period.
- `run_config.json`: exact run settings and period dates.
- `cumulative_returns_after_cost.png`: cumulative performance chart.
- `mean_return_by_k.png`: mean return by portfolio size.
- `lstm_sequence_profile.png`: average top/bottom sequence profile.
- `report_draft.md`: concise text you can adapt into the three-page PDF.

## Improved Price Download

Your original download script already did the most important thing: it used
Yahoo adjusted prices and preserved missing values. The improved downloader adds
metadata and quality checks:

```bash
python3 src/download_sp500_prices.py \
  --start 2000-01-01 \
  --end 2024-12-31 \
  --output-dir data
```

The downloader writes:

- `sp500_prices.csv`: cleaned adjusted close panel.
- `sp500_prices_raw_unfiltered.csv`: raw downloaded panel before sparse-ticker filtering.
- `sp500_constituents.csv`: ticker universe and metadata.
- `sp500_price_quality_report.csv`: missingness, first/last valid date, suspicious-return counts.
- `sp500_download_metadata.json`: reproducibility log and failed/empty tickers.

### Survivorship Bias Warning

The default universe is current S&P 500 constituents. This is convenient, but it
is not survivor-bias-free because firms that left the index before the download
date are missing. Fischer and Krauss use a survivor-bias-free constituent
universe, so this is the largest methodological limitation of the coursework
dataset.

Survivorship bias typically inflates backtested equity-strategy performance,
often by roughly 50-200 bps annualised in US equity studies such as Brown et al.
(1992). Any report based on the default downloader should state that the reported
returns are likely upward-biased. The bias may be especially relevant in the
early subperiod because excluded bankrupt/delisted firms would often have had
poor returns.

For a stronger research design, use a point-in-time historical constituents file:

```bash
python3 src/build_pit_universe.py \
  --start 2000-01-01 \
  --end 2024-12-31 \
  --output data/historical_constituents.csv
```

Then download prices for that historical-union universe:

```bash
python3 src/download_sp500_prices.py \
  --constituents-csv data/historical_constituents.csv \
  --ticker-column Symbol \
  --start 2000-01-01 \
  --end 2024-12-31 \
  --output-dir data
```

This reduces current-constituent survivorship bias, but it is still not a fully
point-in-time tradable universe because it does not enforce daily membership
eligibility in the training/trading sample builders. Treat it as a PIT-union
improvement, not a complete PIT replication.

## Benchmark Comparisons

To compare the LSTM with the paper's benchmark-style models:

```bash
python3 src/compare_benchmark_models.py
```

This runs:

- `LOG`: logistic regression baseline.
- `RAF`: random forest benchmark.
- `DNN`: feed-forward neural network benchmark.
- `Reversal5D`: transparent short-term reversal rule.

It also includes the existing LSTM daily returns from `outputs/daily_portfolio_returns.csv`
when that file exists.

The benchmark script writes results to `outputs_benchmarks/`:

- `benchmark_performance_summary.csv`
- `benchmark_daily_returns.csv`
- `benchmark_cumulative_after_cost.png`
- `benchmark_mean_return_by_k.png`
- `benchmark_report_snippet.md`

For a faster first run:

```bash
python3 src/compare_benchmark_models.py \
  --periods 3 \
  --max-train-samples 60000 \
  --rf-trees 50 \
  --dnn-epochs 8 \
  --log-epochs 8
```

For a fuller run:

```bash
python3 src/compare_benchmark_models.py \
  --periods all \
  --max-train-samples 0 \
  --rf-trees 100 \
  --dnn-epochs 30 \
  --log-epochs 30 \
  --patience 5
```

To run those same benchmarks once across the three subperiods:

```bash
python3 src/compare_benchmark_models.py \
  --csv data/sp500_prices.csv \
  --subperiods decades \
  --periods all \
  --max-train-samples 0 \
  --rf-trees 100 \
  --dnn-epochs 30 \
  --log-epochs 30 \
  --patience 5 \
  --lstm-daily outputs_lstm_subperiods/daily_portfolio_returns_by_subperiod.csv \
  --output-dir outputs_benchmarks_subperiods
```

This writes:

- `outputs_benchmarks_subperiods/benchmark_performance_summary_by_subperiod.csv`
- `outputs_benchmarks_subperiods/benchmark_daily_returns_by_subperiod.csv`
- one detailed output folder per subperiod.

You can also define custom subperiods:

```bash
python3 src/compare_benchmark_models.py \
  --csv data/sp500_prices.csv \
  --subperiods early:2000-01-01:2009-12-31,late:2010-01-01:2024-12-31
```

The original paper uses 1000 random-forest trees. The pure NumPy fallback uses
smaller defaults for runtime. If `scikit-learn` is installed, the script will use
that implementation automatically for the random forest.

## Coursework Framing

The brief does not require exact numerical replication. In the report, be explicit
that the dataset differs from the paper: it starts in 2000 rather than 1990,
extends to 2024 rather than 2015, and defaults to current rather than
point-in-time historical S&P 500 constituents. That means the right comparison is
methodological and directional: whether the rolling LSTM and the long-short
construction behave similarly, whether the signal decays after 2010, and whether
the results survive transaction costs.
