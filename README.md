# Leakage-free index forecasting: deep networks, tuned learners, and naive baselines

Code, data manifest, out-of-sample predictions, and result tables for the paper

> Sukkuea, A., Rungwallapa, T., Tharu, N. K., & Akkajit, P. *Deep Learning Stock Index Forecasting Against Tuned Learners and Naive Baselines Under a Leakage-Free Protocol.* *Machine Learning and Knowledge Extraction* (MDPI), manuscript make-4660090, under review.

The study forecasts the next-day direction of four stock indices over 2022–2025 under one point-in-time contract. SET100, the S&P 500, and CSI 300 are the confirmatory markets; SET50, whose test years an earlier single-market version of this work had used, is exploratory. It asks three questions: (RQ1) do compact deep networks show a significant advantage over naive and classical baselines, (RQ2) does validation-based tuning or a broader search over more recent architectures (GRU, TCN, DLinear, PatchTST-style, MLP) change the comparison, and (RQ3) how much do common protocol errors inflate reported accuracy? The study evaluates directional forecasting performance, not a trading strategy.

![Study design](results/pricevol/figures/study_design.png)

## Main results

| Four-year mean balanced accuracy (%) | SET50 | SET100 | S&P 500 | CSI 300 |
|---|---|---|---|---|
| Best naive rule | 51.5 | 50.2 | 51.2 | 51.9 |
| Best classical learner | 52.4 | 52.7 | 52.6 | 53.6 |
| Best deep network, level regression | 54.3 | 53.0 | 50.1 | 51.8 |
| Best deep network, direct BCE | 50.3 | 50.0 | 50.0 | 51.2 |
| Best tuned classical learner | 52.1 | 52.5 | 50.9 | 54.9 |
| Best tuned deep network | 51.4 | 50.9 | 50.3 | 51.2 |
| Tuned ensemble | 51.9 | 52.4 | 48.7 | 52.4 |
| Best extended network (random search, six architectures) | 53.1 | 52.4 | 50.7 | 52.4 |
| Best network, stationary features | 52.1 | 51.5 | 50.9 | 52.1 |

| Protocol (logistic regression) | SET50 | SET100 | S&P 500 | CSI 300 |
|---|---|---|---|---|
| P0 leak-free reference | 50.2 | 51.2 | 50.6 | 50.6 |
| P1 causal rolling VMD | 49.6 | 51.8 | 52.3 | 53.0 |
| L1 full-sample VMD features | 80.0 | 78.9 | 80.0 | 80.0 |
| L2 full-sample VMD trend as target | 65.4 | 64.0 | 67.4 | 65.4 |
| L3 next-day opening gap as feature | 66.5 | 67.9 | 68.0 | 62.3 |
| L5 L1 + L3 combined | 80.3 | 80.1 | 80.3 | 80.0 |
| L4 "accuracy" = 100 − MAPE (persistence) | 99.3 | 99.3 | 99.2 | 99.2 |

No compact deep network showed a significant advantage over logistic regression or persistence in any confirmatory market, with original or stationary features. Equivalence within ±2 balanced-accuracy points could not be established for any contrast, so small advantages can be neither confirmed nor excluded; the upper 95% confidence limits of the network-minus-baseline differences in the confirmatory markets range from below zero to about 11 points (median about 3). Cross-entropy networks collapsed to predicting almost every session as up (AUC 0.43–0.55).

A tuned random forest on CSI 300 (pooled balanced accuracy 55.0%, 95% CI 52.1–58.0%) performed better than an independence benchmark in the Pesaran–Timmermann test after Holm adjustment, but it showed no clear paired improvement over persistence or logistic regression, did not transfer to the other indices, earned no significant excess return over buy-and-hold after costs, and lost significance with stationary features. It, and the extended LSTM on SET50 and TCN on CSI 300, are candidates for prospective testing, not evidence of dependable forecasting skill. Single protocol errors raised raw accuracy to as much as 80%, and reporting 100 − MAPE gives 99% without any model.

The cost-aware long/flat check uses a proportional cost of 0, 5, or 10 basis points per position change as a stand-in for commissions; spreads, slippage, market impact, and taxes are not modelled, and the Sharpe ratios and maximum drawdowns it reports are descriptive. The results do not establish profitability after realistic transaction costs and do not describe a usable trading strategy.

## Repository layout

```
notebooks/Run_4Markets_colab.ipynb       frozen arm, tuned arm, and leakage audit (Google Colab)
notebooks/Run_Extended_Arm_colab.ipynb   extended arm (LSTM, GRU, TCN, DLinear, PatchTST-style; random search) and revision analyses
notebooks/Run_Editor_Round_colab.ipynb   MLP on the original features and the stationary-feature replication
run_all.py                               runs the frozen, tuned, extended, and leakage code locally
src/benchmark.py                         data alignment, features, naive/classical/deep forecasters, inference, figures
src/tuned.py                             validation-tuned arm (pre-specified grids, class balancing, early stopping)
src/leakage.py                           leakage audit (P0, P1, L1–L5)
src/extended.py                          extended arm (pre-specified 20-configuration random search per architecture, incl. MLP)
src/editor_round.py                      stationary feature set, return-regression networks, and their inference
src/revision2.py                         equivalence tests (TOST), cross-market dependence, cost-aware long/flat check
src/plots.py, src/diagrams.py            additional figures
scripts/verify_data.py                   checks local data files against data/manifest.json (SHA-256)
scripts/editor_analyses.py               AUC, log loss, block-bootstrap AUC intervals, and confidence bounds (Tables A5, A6, A10)
data/manifest.json                       source, instrument id, retrieval date, SHA-256, rows, and date range of every file
results/pricevol/                        original-feature results: tables, tuning and search logs, predictions (no raw prices)
results/pricevol/revision2/              equivalence, dependence, trading, and robustness tables
results/pricevol/editor/                 AUC, log loss, AUC intervals, and confidence bounds
results/stationary/                      stationary-feature replication: skill, contrasts, search log, predictions
```

## Evaluation contract

| Item | Setting |
|---|---|
| Markets | SET100, S&P 500, CSI 300 (confirmatory); SET50 (exploratory); Investing.com, retrieved 2026-10-02 |
| Target | direction of the next session's close; unchanged closes are not scored |
| Test years | 2022, 2023, 2024, 2025 (3,910 scored sessions) |
| Training | expanding window from 2012; last pre-test row purged |
| Weekly / monthly inputs | previous completed bar only |
| Scaling | fitted on training rows only |
| Features | 85 technical variables (75 for S&P 500, which has no volume on Investing.com); stationary set without the 26 price and volume levels: 62 (58) |
| Seeds | 42, 123, 456, 789, 2025 |
| Primary metric | balanced accuracy; AUC and log loss; Pesaran–Timmermann test, circular block bootstrap, exact sign-flip tests, Holm adjustment, minimum detectable effect, equivalence (TOST, ±2 and ±3 points) |
| Extended arm | LSTM, GRU, TCN, DLinear, PatchTST-style, MLP; 20 random configurations each (window 5–60, width 16–128, depth 1–2, dropout, learning rate, weight decay, batch); selected on the last 250 training sessions only |

## Reproduce

### Google Colab
1. Open `notebooks/Run_4Markets_colab.ipynb` in Colab (File → Upload notebook, or open it from GitHub).
2. Put the 12 data files (see below) in `My Drive/Run_4Markets/data/`, or upload them as `Run_4Markets_data.zip` when the notebook asks. All results are written to `My Drive/Run_4Markets/results/`.
3. Runtime → Run all. A CPU runtime is enough; the full study takes roughly 2–3 hours, and finished parts are skipped after a disconnect.
4. Extended arm: open `notebooks/Run_Extended_Arm_colab.ipynb` with the same Drive folder, choose a T4 GPU runtime, and Run all (about 3–5 hours). It reuses the baseline predictions in `results/pricevol/`; if they are missing it refits them, and refitted baselines can differ slightly with other package versions (the paper's inference uses the archived baselines).
5. MLP and stationary features: open `notebooks/Run_Editor_Round_colab.ipynb` with the same Drive folder and Run all on a T4 GPU (about 5–8 hours).
6. AUC, log loss, and confidence bounds: `python scripts/editor_analyses.py results/pricevol`.

### Local
```bash
pip install -r requirements.txt
python scripts/verify_data.py            # checks the 12 CSV files in data/
python run_all.py --quick --skip-leakage # smoke test (a few minutes)
python run_all.py                        # full study; results in results/pricevol/
python run_all.py --price-only           # same study with the 75 price-only features in every market
python run_all.py --skip-frozen --skip-tuned --skip-leakage --extended   # extended arm + revision analyses only
python scripts/editor_analyses.py results/pricevol
```

## Data

Raw prices are **not** included because the provider's terms do not permit redistribution. Download daily, weekly, and monthly historical data for the four indices from the public Investing.com pages listed in `data/manifest.json` (instrument ids 41049, 41047, 166, 940801) for 2012-01-01 to 2026-10-01, save them in `data/` with the file names and columns given in the manifest, and run `scripts/verify_data.py`. Files that differ from the manifest still run, but results may not match exactly. See `data/README.md`.

## Determinism

All models use fixed seeds, deterministic PyTorch algorithms, a single CPU thread for the networks, and unshuffled mini-batches. With the same package versions (`results/pricevol/*/misc.json` records them) the predictions are reproduced exactly; other versions can change individual predictions slightly.

## Licence and citation

Code: MIT licence (`LICENSE`). Result tables, predictions, and figures: CC BY 4.0. Please cite the paper (`CITATION.cff`) if you use this repository.
