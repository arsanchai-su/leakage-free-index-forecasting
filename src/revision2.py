# %% [markdown]
# # Revision analyses from saved predictions (equivalence, cross-market dependence, cost-aware trading, SET50 exclusion)

# %%
DATA, RES = DATA_DIR, OUT_ROOT
OUT = os.path.join(RES, 'revision2'); os.makedirs(OUT, exist_ok=True)
Z90 = stats.norm.ppf(0.95)


# ---------------------------------------------------------------- 1. TOST equivalence
def tost_rows(df, se_col=None):
    rows = []
    for _, r in df.iterrows():
        d = r.delta_daily; se = r[se_col] if se_col else r.mde80 / 2.8
        lo90, hi90 = d - Z90 * se, d + Z90 * se
        out = dict(market=r.market, family=r.family, A=r.A, B=r.B, delta=d, se=se, ci90_lo=lo90, ci90_hi=hi90,
                   min_equiv_margin=max(abs(lo90), abs(hi90)))
        for m in (1.0, 2.0, 3.0):
            p_lower = 1 - stats.norm.cdf((d + m) / se); p_upper = stats.norm.cdf((d - m) / se)
            out[f'p_tost_{int(m)}pp'] = max(p_lower, p_upper)
        rows.append(out)
    return pd.DataFrame(rows)


TB = pd.read_csv(os.path.join(RES, 'all_markets_tableB_contrasts.csv'))
TC = pd.read_csv(os.path.join(RES, 'all_markets_tuned_contrasts.csv'))
eqF = tost_rows(TB[TB.family.isin(['F2 DL vs logistic', 'F3 DL vs persistence'])], 'se_daily')
eqF = eqF.merge(TB[['market', 'A', 'B', 'p_daily_holm']], on=['market', 'A', 'B'])
eqT = tost_rows(TC[TC.family.isin(['T2 tuned vs logistic', 'T3 tuned vs persistence'])])
eqT = eqT.merge(TC[['market', 'A', 'B', 'p_daily_holm']], on=['market', 'A', 'B'])
EQ = pd.concat([eqF.assign(arm='frozen'), eqT.assign(arm='tuned')], ignore_index=True)
for m in (1, 2, 3):
    EQ[f'p_tost_{m}pp_holm'] = EQ.groupby(['market', 'family'])[f'p_tost_{m}pp'].transform(lambda s: holm(s.values))


def verdict(r):
    if r.p_daily_holm < 0.05: return 'different' + (' (A better)' if r.delta > 0 else ' (A worse)')
    if r.p_tost_2pp_holm < 0.05: return 'equivalent within 2 pp'
    if r.p_tost_3pp_holm < 0.05: return 'equivalent within 3 pp only'
    return 'inconclusive'


EQ['verdict'] = EQ.apply(verdict, axis=1)
EQ.to_csv(os.path.join(OUT, 'equivalence_tost.csv'), index=False)

# summary table for the paper: one row per market x comparison group
grp = {'F2 DL vs logistic': 'Frozen DL (LR) vs logistic', 'F3 DL vs persistence': 'Frozen DL (LR) vs persistence',
       'T2 tuned vs logistic': 'Tuned models vs logistic', 'T3 tuned vs persistence': 'Tuned models vs persistence'}
S = []
for (mk, fam), g in EQ.groupby(['market', 'family'], sort=False):
    S.append(dict(market=mk, comparison=grp[fam], n=len(g),
                  n_different=int(g.verdict.str.startswith('different').sum()),
                  n_equiv2=int((g.verdict == 'equivalent within 2 pp').sum()),
                  n_equiv3=int((g.verdict == 'equivalent within 3 pp only').sum()),
                  n_inconclusive=int((g.verdict == 'inconclusive').sum()),
                  margin_range=f'{g.min_equiv_margin.min():.1f}-{g.min_equiv_margin.max():.1f}'))
EQS = pd.DataFrame(S); EQS.to_csv(os.path.join(OUT, 'equivalence_summary.csv'), index=False)

# ---------------------------------------------------------------- 2. cross-market dependence
def daily(mk):
    d = pd.read_csv(os.path.join(DATA, f'{mk}_days.csv'), parse_dates=['Date']).sort_values('Date')
    d['ret'] = d.Close.pct_change(); return d.set_index('Date')


R = pd.DataFrame({mk: daily(mk).ret for mk in MARKETS})
Rt = R[(R.index >= '2022-01-01') & (R.index <= '2025-12-31')]
same = Rt.dropna()
# Asian sessions on day t react to the US close of day t-1 (US closes after Asian markets)
Rl = Rt.copy(); spx = R.SPX.dropna(); Rl['SPX'] = spx.shift(1).reindex(Rl.index, method=None)
Rl['SPX'] = spx.reindex(spx.index).shift(1).reindex(Rl.index)
lag = Rl.dropna()
C_same = same.corr(); C_lag = lag.corr()
Dir = (same > 0).astype(int); Phi = Dir.corr()


def n_eff(C):
    ev = np.linalg.eigvalsh(C.values); return float(ev.sum() ** 2 / (ev ** 2).sum())


dep = dict(n_common_sessions=len(same), n_common_sessions_lagged=len(lag),
           corr_same_day=C_same.round(3).to_dict(), corr_us_lagged=C_lag.round(3).to_dict(), phi_direction_same_day=Phi.round(3).to_dict(),
           n_eff_same_day=n_eff(C_same), n_eff_us_lagged=n_eff(C_lag))
# yearly agreement of the sign of model-minus-baseline differences across markets
json.dump(dep, open(os.path.join(OUT, 'cross_market_dependence.json'), 'w'), indent=1)

# ---------------------------------------------------------------- 3. trading evaluation
def load_preds(mk):
    pdir = os.path.join(RES, mk, 'predictions')
    return pd.concat([pd.read_csv(f, parse_dates=['Date']) for f in sorted(glob.glob(os.path.join(pdir, 'predictions_*.csv')))], ignore_index=True)


def ens(allp, m):
    g = allp[allp.method == m]
    s = g.groupby('Date').agg(score=('score', 'mean'), year=('year', 'first'))
    fam = g.family.iloc[0]
    if m == 'Train-majority': s['pred'] = g.groupby('Date').pred.first()
    elif fam == 'Naive' or m == 'Ridge-return' or fam == 'DL-R': s['pred'] = (s.score > 0).astype(int)
    else: s['pred'] = (s.score > 0.5).astype(int)
    return s.sort_index()


def perf(pos, r, cost_bps):
    trades = np.abs(np.diff(np.r_[0, pos]))
    net = pos * r - trades * cost_bps / 1e4
    ann = 252
    mu = net.mean() * ann; vol = net.std(ddof=1) * np.sqrt(ann)
    eq = np.cumprod(1 + net); mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return net, dict(ann_ret=100 * ((eq[-1]) ** (ann / len(net)) - 1), ann_vol=100 * vol, sharpe=mu / vol if vol > 0 else np.nan,
                     max_dd=100 * mdd, trades_per_year=trades.sum() / len(net) * ann, time_in_market=100 * pos.mean())


MODELS = ['Persistence', 'MA20-trend', 'Logistic-L2', 'Random forest', 'LightGBM', 'LSTM (level-regression)',
          'LSTM-Attention (level-regression)', 'Logistic-L2 (tuned)', 'Random forest (tuned)', 'LightGBM (tuned)',
          'LSTM-Attention (tuned)', 'Ensemble (tuned)']
if os.path.exists(os.path.join(RES, MARKETS[0], 'predictions', 'predictions_ext_LSTM.csv')):
    MODELS += [f'{a} (extended)' for a in EXT_ARCHS]
rng = np.random.default_rng(20261022)
rows = []
for mk in MARKETS:
    allp = load_preds(mk); d = daily(mk)
    d['ret_next'] = d.Target_Next_Close / d.Close - 1
    base = ens(allp, 'Logistic-L2'); dates = base.index
    r = d.ret_next.reindex(dates).values; assert not np.isnan(r).any()
    n = len(r); k = int(np.ceil(n / BLOCK)); starts = rng.integers(0, n, size=(B, k))
    IDX = ((starts[:, :, None] + np.arange(BLOCK)[None, None, :]) % n).reshape(B, -1)[:, :n]
    bh_net, bh = perf(np.ones(n), r, 0)
    rows.append(dict(market=mk, model='Buy-and-hold', cost_bps=0, **bh, excess_ann=0.0, ex_lo=np.nan, ex_hi=np.nan, p_excess_gt0=np.nan))
    for m in [m_ for m_ in MODELS if m_ in set(allp.method)]:
        pos = ens(allp, m).pred.reindex(dates).values.astype(float)
        for c in (0, 5, 10):
            net, st = perf(pos, r, c)
            ex = net - bh_net; bs = ex[IDX].mean(1) * 252 * 100
            rows.append(dict(market=mk, model=m, cost_bps=c, **st, excess_ann=ex.mean() * 252 * 100,
                             ex_lo=np.percentile(bs, 2.5), ex_hi=np.percentile(bs, 97.5), p_excess_gt0=float(np.mean(bs <= 0))))
TR = pd.DataFrame(rows)
TR['p_holm'] = TR.groupby(['market', 'cost_bps']).p_excess_gt0.transform(lambda q: holm(q.fillna(1).values) if q.notna().any() else q)
TR.loc[TR.model == 'Buy-and-hold', 'p_holm'] = np.nan
TR.to_csv(os.path.join(OUT, 'trading_long_flat.csv'), index=False)

# ---------------------------------------------------------------- 4. multiplicity without SET50
TT = pd.read_csv(os.path.join(RES, 'all_markets_tuned_skill.csv'))
sub = TT[TT.market != 'SET50'].copy(); sub['pt_holm_excl_set50'] = holm(sub.pt_p.fillna(1).values)
TT['pt_holm_all36'] = holm(TT.pt_p.fillna(1).values)
TA = pd.read_csv(os.path.join(RES, 'all_markets_tableA_skill.csv'))
rob = dict(rf_csi300_pt_p=float(TT[(TT.market == 'CSI300') & (TT.method == 'Random forest (tuned)')].pt_p.iloc[0]),
           rf_csi300_holm_all36=float(TT[(TT.market == 'CSI300') & (TT.method == 'Random forest (tuned)')].pt_holm_all36.iloc[0]),
           rf_csi300_holm_excl_set50=float(sub[(sub.market == 'CSI300') & (sub.method == 'Random forest (tuned)')].pt_holm_excl_set50.iloc[0]),
           n_tuned_excl_set50=len(sub),
           frozen_min_pt_holm_by_market=TA.groupby('market').pt_holm.min().round(3).to_dict(),
           frozen_dl_lr_beats_logistic_any_excl_set50=bool(((TB.family == 'F2 DL vs logistic') & (TB.market != 'SET50') & (TB.p_daily_holm < 0.05) & (TB.delta_daily > 0)).any()))
json.dump(rob, open(os.path.join(OUT, 'robustness_excl_set50.json'), 'w'), indent=1)

print(EQS.to_string()); print(EQ.verdict.value_counts())
print(json.dumps(dep, indent=1)); print(json.dumps(rob, indent=1))
print(TR[TR.cost_bps.isin([0, 10])].round(2).to_string())
