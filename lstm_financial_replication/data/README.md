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
  --output data/historical_constituents.csv

python3 src/download_sp500_prices.py \
  --constituents-csv data/historical_constituents.csv \
  --ticker-column Symbol \
  --start 2000-01-01 \
  --end 2024-12-31 \
  --output-dir data
```

This reduces current-constituent survivorship bias, but it is still not a fully
point-in-time trading universe because it does not enforce daily membership
eligibility. For a stricter replication, provide a daily point-in-time
constituent source and extend the model pipeline with a membership mask.

You can also provide your own constituents file:

```bash
python3 src/download_sp500_prices.py \
  --constituents-csv historical_constituents.csv \
  --ticker-column Symbol \
  --output-dir data
```
