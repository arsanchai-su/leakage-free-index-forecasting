# Data

Raw index prices are not redistributed. Place these 12 files in this folder:

| Market | Investing.com instrument | Files |
|---|---|---|
| SET50 | 41049 | SET50_days.csv, SET50_weeks.csv, SET50_months.csv |
| SET100 | 41047 | SET100_days.csv, SET100_weeks.csv, SET100_months.csv |
| S&P 500 | 166 | SPX_days.csv, SPX_weeks.csv, SPX_months.csv |
| CSI 300 | 940801 | CSI300_days.csv, CSI300_weeks.csv, CSI300_months.csv |

Columns: `Date,Close,Open,High,Low,Volume,Change_pct,Target_Next_Close` with ISO dates in ascending order. `Target_Next_Close` is the next bar's close (empty on the last row); the code recomputes it from `Close`. Weekly bars are dated on the Sunday that starts the week and monthly bars on the first day of the month, as exported by Investing.com. Volume is empty for the S&P 500.

Period used in the paper: 2012-01-01 to 2026-10-01 (analysis ends 2025-12-31). Run `python scripts/verify_data.py` to compare your files with the SHA-256 digests in `manifest.json`.
