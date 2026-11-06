# %% [markdown]
# # Editor round: MLP baseline and stationary-feature replication
#
# Part A adds an MLP to the extended arm on the original feature set (results/pricevol).
# Part B repeats the main comparison with STATIONARY features only (results/stationary): price-level variables (raw
# OHLC, close and log-volume lags and levels, moving-average levels, momentum in index points, MACD in index points)
# are removed, MACD is re-expressed relative to the close, and the regression objective of the frozen networks predicts
# the next-day RETURN instead of the next close level. Everything else (folds, purge, train-only scaling, seeds,
# validation windows, search space, inference) is unchanged.

# %%
from sklearn.metrics import roc_auc_score, log_loss
LEVEL_VARS = (['Close_D', 'Open_D', 'High_D', 'Low_D', 'logVol_D'] + [f'close_lag{k}' for k in range(1, 6)]
              + [f'logvol_lag{k}' for k in range(1, 6)] + [f'ma_{k}' for k in (5, 10, 20, 60)]
              + [f'mom_{k}' for k in (5, 10, 20, 60)] + ['macd', 'macd_sig', 'macd_hist'])


def stationary_features(D, FEATS):
    D = D.copy()
    for c in ['macd', 'macd_sig', 'macd_hist']: D[c + '_rel'] = D[c] / D.C
    feats = [f for f in FEATS if f not in LEVEL_VARS] + ['macd_rel', 'macd_sig_rel', 'macd_hist_rel']
    return D, feats


def run_dl_stationary(D, FEATS, arch, path, mk):
    """Frozen networks with return regression (RR) instead of level regression, plus direct BCE."""
    if os.path.exists(path): return
    X_all, W, out = D[FEATS].values, WIN[arch], []
    for year, s, e in FOLDS:
        tr_idx = np.where(D.Date < s)[0][:-1]; tr_idx = tr_idx[tr_idx >= W - 1]
        te_idx = np.where((D.Date >= s) & (D.Date <= e))[0]
        if len(te_idx) == 0: continue
        sx = MinMaxScaler().fit(X_all[tr_idx]); Xs = sx.transform(X_all)
        sy = MinMaxScaler().fit(D.ret_next.values[tr_idx, None])
        Xtr, Xte = seqs(Xs, tr_idx, W), seqs(Xs, te_idx, W)
        targets = {'R': sy.transform(D.ret_next.values[tr_idx, None]).astype(np.float32), 'C': D.y.values[tr_idx, None].astype(np.float32)}
        te = D.iloc[te_idx]
        for obj in ['R', 'C']:
            for sd in SEEDS:
                torch.manual_seed(sd); np.random.seed(sd)
                net = Net(arch, X_all.shape[1]); opt = torch.optim.Adam(net.parameters(), 1e-3)
                lossf = nn.MSELoss() if obj == 'R' else nn.BCEWithLogitsLoss()
                xt, yt = torch.from_numpy(Xtr), torch.from_numpy(targets[obj])
                net.train()
                for ep in range(EPOCHS):
                    for b in range(0, len(xt), 32):
                        opt.zero_grad(); loss = lossf(net(xt[b:b+32]), yt[b:b+32]); loss.backward(); opt.step()
                net.eval()
                with torch.no_grad(): o = net(torch.from_numpy(Xte)).numpy().ravel()
                if obj == 'R': score = sy.inverse_transform(o[:, None]).ravel(); pred = score > 0
                else: score = 1/(1 + np.exp(-o)); pred = score > 0.5
                out.append(frame(f'{arch} ({"return-regression" if obj == "R" else "direct-BCE"})', 'DL-' + obj, sd, year, te, pred, score, arch=arch))
            print(f'{mk:7s} {arch:20s} {year} {obj} done', flush=True)
    pd.concat(out).to_csv(path, index=False)


def ensure_baselines(D, FEATS, pred_dir, mk):
    if not os.path.exists(os.path.join(pred_dir, 'predictions_classical.csv')):
        run_classical(D, FEATS, os.path.join(pred_dir, 'predictions_classical.csv'))
    if not os.path.exists(os.path.join(pred_dir, 'predictions_tuned_classical.csv')):
        run_classical_tuned(D, FEATS, os.path.join(pred_dir, 'predictions_tuned_classical.csv'), mk, [])

# %%
# ---------------- Part A: MLP on the original features ----------------
for mk in EXT_MARKETS:
    out_dir = os.path.join(OUT_ROOT, mk); pred_dir = os.path.join(out_dir, 'predictions'); os.makedirs(pred_dir, exist_ok=True)
    A = build_aligned(mk); has_vol = A['Volume_D'].notna().all() and (A['Volume_D'] > 0).all()
    D, FEATS = build_features(A, has_vol if USE_VOLUME == 'auto' else False)
    ensure_baselines(D, FEATS, pred_dir, mk)
    run_extended(D, FEATS, 'MLP', os.path.join(pred_dir, 'predictions_ext_MLP.csv'), mk, os.path.join(OUT_ROOT, 'extended_search_log_MLP.csv'))
    print(mk, 'MLP (original features) done', flush=True)

# %%
# ---------------- Part B: stationary-feature replication ----------------
OUT_S = os.path.join(ROOT, 'results', ('quick_' if QUICK else '') + 'stationary')
STAT_ARCHS = ['LSTM', 'GRU', 'TCN', 'DLinear', 'PatchTST', 'MLP']
t_s = time.time()
for mk in EXT_MARKETS:
    out_dir = os.path.join(OUT_S, mk); pred_dir = os.path.join(out_dir, 'predictions'); os.makedirs(pred_dir, exist_ok=True)
    A = build_aligned(mk); has_vol = A['Volume_D'].notna().all() and (A['Volume_D'] > 0).all()
    D0, F0 = build_features(A, has_vol if USE_VOLUME == 'auto' else False)
    D, FEATS = stationary_features(D0, F0)
    print(f'{mk}: {len(FEATS)} stationary features', flush=True)
    ensure_baselines(D, FEATS, pred_dir, mk)
    for arch in ARCHS:
        run_dl_stationary(D, FEATS, arch, os.path.join(pred_dir, f'predictions_dl_{arch}.csv'), mk)
    for arch in STAT_ARCHS:
        run_extended(D, FEATS, arch, os.path.join(pred_dir, f'predictions_ext_{arch}.csv'), mk, os.path.join(OUT_S, 'extended_search_log.csv'))
    json.dump(dict(market=mk, n_features=len(FEATS), features=FEATS), open(os.path.join(out_dir, 'features.json'), 'w'), indent=1)
    print(f'{mk} stationary replication done, elapsed {(time.time() - t_s)/60:.1f} min', flush=True)

# %%
# ---------------- inference for the stationary replication (same tests as the main analysis) ----------------
Z90 = stats.norm.ppf(0.95)


def ens_scores(allp, m):
    g = allp[allp.method == m]
    s = g.groupby('Date').agg(y=('y', 'first'), score=('score', 'mean'), year=('year', 'first'), q=('q', 'first'))
    fam = g.family.iloc[0]
    if m == 'Train-majority': s['pred'] = g.groupby('Date').pred.first()
    elif fam in ('Naive', 'DL-R') or m == 'Ridge-return': s['pred'] = (s.score > 0).astype(int)
    else: s['pred'] = (s.score > 0.5).astype(int)
    return s.sort_index(), fam


def generic_inference(mk, out_dir):
    rng = np.random.default_rng(20260926)
    allp = pd.concat([pd.read_csv(f, parse_dates=['Date']) for f in sorted(glob.glob(os.path.join(out_dir, 'predictions', 'predictions_*.csv')))], ignore_index=True)
    allp = allp[~allp.zero].copy(); allp['q'] = allp.Date.dt.year.astype(str) + 'Q' + allp.Date.dt.quarter.astype(str)
    methods = list(dict.fromkeys(allp.method)); E = {m: ens_scores(allp, m) for m in methods}
    n = len(E['Logistic-L2'][0]); k = int(np.ceil(n / BLOCK)); starts = rng.integers(0, n, size=(B, k))
    IDX = ((starts[:, :, None] + np.arange(BLOCK)[None, None, :]) % n).reshape(B, -1)[:, :n]
    rows = []
    for m in methods:
        e, fam = E[m]; g = allp[allp.method == m]
        per = g.groupby(['year', 'seed']).apply(lambda t: bacc(t.y.values, t.pred.values)).groupby('year').mean()
        y, p = e.y.values, e.pred.values; bb = boot_bacc(y, p, IDX); s_, ppt = pt_test(y, p)
        auc = roc_auc_score(y, e.score.values) if m != 'Train-majority' else 0.5
        prob = fam in ('Classical', 'DL-C', 'Tuned', 'Extended') and m != 'Ridge-return'
        ll = log_loss(y, np.clip(e.score.values, 1e-6, 1 - 1e-6)) if prob else np.nan
        rows.append(dict(market=mk, method=m, family=fam, **{f'bacc_{yr}': 100*v for yr, v in per.items()}, bacc_mean=100*per.mean(),
                         bacc_pooled=100*bacc(y, p), ci_lo=100*np.nanpercentile(bb, 2.5), ci_hi=100*np.nanpercentile(bb, 97.5),
                         up_pred=100*p.mean(), auc=auc, logloss=ll, pt_p=ppt, n=len(y)))
    TS = pd.DataFrame(rows); TS['pt_holm'] = holm(TS.pt_p.fillna(1).values)

    def ub(m, unit):
        g = allp[allp.method == m]
        return g.groupby([unit, 'seed']).apply(lambda t: bacc(t.y.values, t.pred.values)).groupby(unit).mean()

    def con(a, b, fam):
        dy = (ub(a, 'year') - ub(b, 'year')).values; dq = (ub(a, 'q') - ub(b, 'q')).values
        ea, eb = E[a][0], E[b][0]; y = ea.y.values
        dd = boot_bacc(y, ea.pred.values, IDX) - boot_bacc(y, eb.pred.values, IDX)
        obs = bacc(y, ea.pred.values) - bacc(y, eb.pred.values); se = np.nanstd(dd)
        tost = max(1 - stats.norm.cdf((obs + 0.02)/se), stats.norm.cdf((obs - 0.02)/se)) if se > 0 else 1.0
        return dict(market=mk, family=fam, A=a, B=b, delta_daily=100*obs, se_daily=100*se, ci_lo=100*(obs - 1.96*se), ci_hi=100*(obs + 1.96*se),
                    p_daily=float(2*(1 - stats.norm.cdf(abs(obs)/se))) if se > 0 else 1.0, p_quarter=signflip_p(dq), p_year=signflip_p(dy),
                    mde80=100*2.8*se, min_equiv_margin=100*(abs(obs) + Z90*se), p_tost_2pp=tost)
    deep = [m for m in methods if E[m][1] in ('DL-R', 'Extended')]
    R = [con(m, 'Logistic-L2', 'S1 deep vs logistic') for m in deep] + [con(m, 'Persistence', 'S2 deep vs persistence') for m in deep]
    TC = pd.DataFrame(R)
    for c in ['p_year', 'p_quarter', 'p_daily', 'p_tost_2pp']:
        TC[c + '_holm'] = TC.groupby('family')[c].transform(lambda s: holm(s.values))
    TS.to_csv(os.path.join(out_dir, 'tableS_skill.csv'), index=False); TC.to_csv(os.path.join(out_dir, 'tableS_contrasts.csv'), index=False)
    return TS, TC


SS, SC = [], []
for mk in EXT_MARKETS:
    a, b = generic_inference(mk, os.path.join(OUT_S, mk)); SS.append(a); SC.append(b)
SS, SC = pd.concat(SS), pd.concat(SC)
SS.to_csv(os.path.join(OUT_S, 'all_markets_stationary_skill.csv'), index=False); SC.to_csv(os.path.join(OUT_S, 'all_markets_stationary_contrasts.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print('Stationary features: four-year mean balanced accuracy (%)')
print(SS.pivot_table(index='method', columns='market', values='bacc_mean', sort=False)[EXT_MARKETS].round(2).to_string())
print('\nAUC'); print(SS.pivot_table(index='method', columns='market', values='auc', sort=False)[EXT_MARKETS].round(3).to_string())
print('\nPredicted up share (%)'); print(SS.pivot_table(index='method', columns='market', values='up_pred', sort=False)[EXT_MARKETS].round(1).to_string())
sig = SC[(SC.p_daily_holm < 0.05) | (SC.p_quarter_holm < 0.05)]
print('\nDeep vs baseline contrasts significant after Holm:'); print(sig[['market', 'family', 'A', 'delta_daily', 'p_daily_holm', 'p_quarter_holm', 'p_year_holm']].round(3).to_string() if len(sig) else 'none')

# %%
# ---------------- download (no raw prices) ----------------
import zipfile as _zf
zip_path = os.path.join(ROOT, f'results_editor_round{"_quick" if QUICK else ""}.zip')   # saved in Google Drive (Run_4Markets)
with _zf.ZipFile(zip_path, 'w', _zf.ZIP_DEFLATED) as z:
    for root_, _, fs in os.walk(OUT_S):
        for f in fs: z.write(os.path.join(root_, f), os.path.relpath(os.path.join(root_, f), os.path.dirname(OUT_S)))
    for mk in EXT_MARKETS:
        f = os.path.join(OUT_ROOT, mk, 'predictions', 'predictions_ext_MLP.csv')
        if os.path.exists(f): z.write(f, os.path.relpath(f, os.path.dirname(OUT_ROOT)))
    f = os.path.join(OUT_ROOT, 'extended_search_log_MLP.csv')
    if os.path.exists(f): z.write(f, os.path.relpath(f, os.path.dirname(OUT_ROOT)))
print('created', zip_path, '(in Google Drive)' if ROOT.startswith('/content/drive') else '(local runtime only: download it now)')
try:
    from google.colab import files; files.download(zip_path)
except Exception:
    pass
