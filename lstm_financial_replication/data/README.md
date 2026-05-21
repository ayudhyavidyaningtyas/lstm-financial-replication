# Data Directory

Generated data files are intentionally not committed. Build the price panel with:

```bash
python3 src/download_sp500_prices.py --output-dir data
```

The default scripts expect:

```text
data/sp500_prices.csv
```

The default downloader uses current S&P 500 constituents, which is convenient
but not survivor-bias-free. A better stopgap is to build a historical-union
constituent file, then download prices for that wider universe:

```bash
python3 src/build_pit_universe.py \
  --start 2000-01-01 \
  --end 2024-12-31 \
  --output data/historical_constituents.csv \
  --snapshots data/sp500_membership_snapshots.csv

python3 src/download_sp500_prices.py \
  --constituents-csv data/historical_constituents.csv \
  --ticker-column Symbol \
  --start 2000-01-01 \
  --end 2024-12-31 \
  --output-dir data
```

This reduces current-constituent survivorship bias. Pass the generated
`sp500_membership_snapshots.csv` file to the model scripts with
`--membership-csv` to restrict samples, rankings, and target medians to stocks
that were members on the signal date. Residual limitations remain because Yahoo
may not provide complete delisting returns.

You can also provide your own constituents file:

```bash
python3 src/download_sp500_prices.py \
  --constituents-csv historical_constituents.csv \
  --ticker-column Symbol \
  --output-dir data
```
