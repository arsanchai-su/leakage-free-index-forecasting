# %% [markdown]
# # Tuned arm (pre-specified before any test-year result was seen for these settings)
#
# Every choice below is made on a validation window cut from the END of each training set (last 250 sessions, with a
# one-row purge). The 2022-2025 test years are touched once, after selection. Changes relative to the frozen protocol:
#
# * Target: sign of the next-day return (instead of the next close level)
# * Class balance: `class_weight='balanced'` for classical learners and `pos_weight = n_down/n_up` for networks,
#   so the 0.5 threshold is meaningful and majority-side collapse is discouraged
# * Classical: small grids selected by validation balanced accuracy, then refit on the full training window
# * Networks: standardized inputs, hidden size 32, dropout 0.2, weight decay 1e-4, early stopping on validation balanced
#   accuracy (max 30 epochs, patience 5); refit on the full training window for the selected number of epochs
# * Ensemble (fixed in advance): mean probability of the three tuned classical learners and the five tuned networks

# %%
TUNED_MARKETS = [m for m in os.environ.get('TUNED_MARKETS', ','.join(MARKETS)).split(',') if m]
VAL_N, MAX_EP, PATIENCE = 250, (3 if QUICK else 30), (1 if QUICK else 5)
GRID = {
    'Logistic-L2 (tuned)': [dict(C=c) for c in (0.01, 0.1, 1.0)],
    'Random forest (tuned)': [dict(min_samples_leaf=m, max_depth=d) for m in (20, 50, 100) for d in (None, 5)],
    'LightGBM (tuned)': [dict(num_leaves=nl, n_estimators=ne) for nl in (7, 15) for ne in (100, 300)],
}


def make_clf(name, prm, sd):
    if name.startswith('Logistic'):
        return LogisticRegression(C=prm['C'], max_iter=5000, class_weight='balanced')
    if name.startswith('Random'):
        return RandomForestClassifier(n_estimators=500, max_features='sqrt', n_jobs=-1, random_state=sd, class_weight='balanced', **prm)
    return lgb.LGBMClassifier(learning_rate=0.03, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              class_weight='balanced', random_state=sd, verbose=-1, **prm)


def split_val(D, s):
    tr = D[D.Date < s].iloc[:-1]                       # outer purge
    fit, val = tr.iloc[:-(VAL_N + 1)], tr.iloc[-VAL_N:]  # inner purge of one row
    return tr, fit, val


def run_classical_tuned(D, FEATS, path, mk, log):
    if os.path.exists(path): return
    rows = []
    for year, s, e in FOLDS:
        tr, fit, val = split_val(D, s); te = D[(D.Date >= s) & (D.Date <= e)]
        if te.empty: continue
        sc_f = StandardScaler().fit(fit[FEATS]); sc = StandardScaler().fit(tr[FEATS])
        for name, grid in GRID.items():
            seeds = [0] if name.startswith('Logistic') else SEEDS
            def vscore(prm):
                v = bacc(val.y.values, make_clf(name, prm, seeds[0]).fit(sc_f.transform(fit[FEATS]), fit.y.values).predict(sc_f.transform(val[FEATS])))
                return -1 if np.isnan(v) else v
            best = max(grid, key=vscore)
            log.append(dict(market=mk, year=year, method=name, chosen=str(best)))
            for sd in seeds:
                p = make_clf(name, best, sd).fit(sc.transform(tr[FEATS]), tr.y.values).predict_proba(sc.transform(te[FEATS]))[:, 1]
                rows.append(frame(name, 'Tuned', sd, year, te, p > 0.5, p))
    pd.concat(rows).to_csv(path, index=False)


class NetT(Net):
    def __init__(s, arch, nf, h=32):
        nn.Module.__init__(s); s.arch = arch; s.drop = nn.Dropout(0.2)
        if 'LSTM' in arch: s.lstm = nn.LSTM(nf, h, batch_first=True)
        dim = h if 'LSTM' in arch else nf
        if 'CNN' in arch: s.conv = CausalConv(dim, 32); dim = 32
        if 'Attention' in arch: s.att = CausalAtt(dim)
        s.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(dim, 8), nn.ReLU(), nn.Linear(8, 1))


def train_net(arch, nf, Xtr, ytr, sd, epochs=None, Xval=None, yval=None):
    torch.manual_seed(sd); np.random.seed(sd)
    net = NetT(arch, nf); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    pw = torch.tensor([(ytr == 0).sum()/max((ytr == 1).sum(), 1)], dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr[:, None].astype(np.float32))
    best, best_ep, bad = -1, 1, 0
    for ep in range(1, (epochs or MAX_EP) + 1):
        net.train()
        for b in range(0, len(xt), 32):
            opt.zero_grad(); loss = lossf(net(xt[b:b+32]), yt[b:b+32]); loss.backward(); opt.step()
        if epochs is None:
            net.eval()
            with torch.no_grad(): pv = net(torch.from_numpy(Xval)).numpy().ravel() > 0
            sc_ = bacc(yval, pv.astype(int))
            sc_ = -1 if np.isnan(sc_) else sc_
            if sc_ > best + 1e-9: best, best_ep, bad = sc_, ep, 0
            else:
                bad += 1
                if bad >= PATIENCE: break
    return net, best_ep, best


def run_dl_tuned(D, FEATS, arch, path, mk, log):
    if os.path.exists(path): return
    X_all, W, out = D[FEATS].values, WIN[arch], []
    for year, s, e in FOLDS:
        tr_idx = np.where(D.Date < s)[0][:-1]; tr_idx = tr_idx[tr_idx >= W - 1]
        fit_idx, val_idx = tr_idx[:-(VAL_N + 1)], tr_idx[-VAL_N:]
        te_idx = np.where((D.Date >= s) & (D.Date <= e))[0]
        if len(te_idx) == 0: continue
        te = D.iloc[te_idx]; y = D.y.values
        for sd in SEEDS:
            t0 = time.time()
            sf = StandardScaler().fit(X_all[fit_idx]); Xf = np.clip(sf.transform(X_all), -10, 10)
            _, ep, vb = train_net(arch, X_all.shape[1], seqs(Xf, fit_idx, W), y[fit_idx], sd, None, seqs(Xf, val_idx, W), y[val_idx])
            st = StandardScaler().fit(X_all[tr_idx]); Xs = np.clip(st.transform(X_all), -10, 10)
            net, _, _ = train_net(arch, X_all.shape[1], seqs(Xs, tr_idx, W), y[tr_idx], sd, epochs=ep)
            net.eval()
            with torch.no_grad(): o = net(torch.from_numpy(seqs(Xs, te_idx, W))).numpy().ravel()
            p = 1/(1 + np.exp(-o))
            out.append(frame(f'{arch} (tuned)', 'Tuned', sd, year, te, p > 0.5, p, arch=arch))
            log.append(dict(market=mk, year=year, method=f'{arch} (tuned)', seed=sd, epochs=ep, val_bacc=vb))
            print(f'{mk:7s} {arch:20s} {year} tuned seed {sd:4d} ep {ep:2d} val {vb:.3f} {time.time()-t0:5.1f}s', flush=True)
    pd.concat(out).to_csv(path, index=False)


def ensemble_tuned(pred_dir):
    path = os.path.join(pred_dir, 'predictions_tuned_ensemble.csv')
    parts = [pd.read_csv(os.path.join(pred_dir, 'predictions_tuned_classical.csv'), parse_dates=['Date'])]
    parts += [pd.read_csv(os.path.join(pred_dir, f'predictions_tuned_dl_{a}.csv'), parse_dates=['Date']) for a in ARCHS]
    g = pd.concat(parts).groupby(['method', 'Date']).agg(y=('y', 'first'), zero=('zero', 'first'), year=('year', 'first'), score=('score', 'mean')).reset_index()
    e = g.groupby('Date').agg(y=('y', 'first'), zero=('zero', 'first'), year=('year', 'first'), score=('score', 'mean')).reset_index()
    e['method'], e['family'], e['seed'], e['pred'] = 'Ensemble (tuned)', 'Tuned', 0, (e.score > 0.5).astype(int)
    e[['method', 'family', 'seed', 'year', 'Date', 'y', 'zero', 'pred', 'score']].to_csv(path, index=False)


def inference_tuned(mk, out_dir):
    rng = np.random.default_rng(20260926)
    pred_dir = os.path.join(out_dir, 'predictions')
    allp = pd.concat([pd.read_csv(f, parse_dates=['Date']) for f in sorted(glob.glob(os.path.join(pred_dir, 'predictions_*.csv')))], ignore_index=True)
    allp = allp[~allp.zero].copy(); allp['q'] = allp.Date.dt.year.astype(str) + 'Q' + allp.Date.dt.quarter.astype(str)
    tuned = list(GRID) + [f'{a} (tuned)' for a in ARCHS] + ['Ensemble (tuned)']
    frozen = {'Logistic-L2 (tuned)': 'Logistic-L2', 'Random forest (tuned)': 'Random forest', 'LightGBM (tuned)': 'LightGBM'}
    frozen.update({f'{a} (tuned)': f'{a} (level-regression)' for a in ARCHS})

    def ens(m):
        g = allp[allp.method == m]
        s = g.groupby('Date').agg(y=('y', 'first'), score=('score', 'mean'), year=('year', 'first'), q=('q', 'first'))
        fam = g.family.iloc[0]
        if m == 'Train-majority': s['pred'] = g.groupby('Date').pred.first()
        elif fam == 'Naive' or m == 'Ridge-return' or fam == 'DL-R': s['pred'] = (s.score > 0).astype(int)
        else: s['pred'] = (s.score > 0.5).astype(int)
        return s.sort_index()

    names = tuned + sorted(set(frozen.values())) + ['Persistence', 'Train-majority']
    E = {m: ens(m) for m in names}
    n = len(E[tuned[0]]); k = int(np.ceil(n/BLOCK)); starts = rng.integers(0, n, size=(B, k))
    IDX = ((starts[:, :, None] + np.arange(BLOCK)[None, None, :]) % n).reshape(B, -1)[:, :n]
    rows = []
    for m in tuned:
        g = allp[allp.method == m]
        per = g.groupby(['year', 'seed']).apply(lambda t: bacc(t.y.values, t.pred.values)).groupby('year').mean()
        y, p = E[m].y.values, E[m].pred.values; bb = boot_bacc(y, p, IDX); s_, ppt = pt_test(y, p)
        rows.append(dict(market=mk, method=m, **{f'bacc_{yr}': 100*v for yr, v in per.items()}, bacc_mean=100*per.mean(),
                         bacc_pooled=100*bacc(y, p), ci_lo=100*np.nanpercentile(bb, 2.5), ci_hi=100*np.nanpercentile(bb, 97.5),
                         acc=100*(y == p).mean(), mcc=mcc(y, p), up_pred=100*p.mean(), pt_p=ppt, n=len(y)))
    TT = pd.DataFrame(rows); TT['pt_holm'] = holm(TT.pt_p.fillna(1).values)

    def ub(m, unit):
        g = allp[allp.method == m]
        return g.groupby([unit, 'seed']).apply(lambda t: bacc(t.y.values, t.pred.values)).groupby(unit).mean()

    def con(a, b, fam):
        dy = (ub(a, 'year') - ub(b, 'year')).values
        y = E[a].y.values; dd = boot_bacc(y, E[a].pred.values, IDX) - boot_bacc(y, E[b].pred.values, IDX)
        obs = bacc(y, E[a].pred.values) - bacc(y, E[b].pred.values); se = np.nanstd(dd)
        return dict(market=mk, family=fam, A=a, B=b, delta_year_mean=100*dy.mean(), p_year=signflip_p(dy), delta_daily=100*obs,
                    p_daily=float(2*(1 - stats.norm.cdf(abs(obs)/se))) if se > 0 else 1.0, mde80=100*2.8*se)

    R = [con(m, frozen[m], 'T1 tuned vs frozen') for m in frozen]
    R += [con(m, 'Logistic-L2', 'T2 tuned vs logistic') for m in tuned if m != 'Logistic-L2 (tuned)']
    R += [con(m, 'Persistence', 'T3 tuned vs persistence') for m in tuned]
    TC = pd.DataFrame(R)
    for c in ['p_year', 'p_daily']: TC[c + '_holm'] = TC.groupby('family')[c].transform(lambda s: holm(s.values))
    TT.to_csv(os.path.join(out_dir, 'tableT_tuned_skill.csv'), index=False); TC.to_csv(os.path.join(out_dir, 'tableT_tuned_contrasts.csv'), index=False)
    return TT, TC

# %%
TUNE_LOG = []
for mk in TUNED_MARKETS:
    out_dir = os.path.join(OUT_ROOT, mk); pred_dir = os.path.join(out_dir, 'predictions'); os.makedirs(pred_dir, exist_ok=True)
    A = build_aligned(mk)
    has_vol = A['Volume_D'].notna().all() and (A['Volume_D'] > 0).all()
    use_vol = has_vol if USE_VOLUME == 'auto' else False
    D, FEATS = build_features(A, use_vol)
    D['y'] = (D.C_next > D.C).astype(int)            # sign of next-day return (same as before; kept explicit)
    if not os.path.exists(os.path.join(pred_dir, 'predictions_classical.csv')):
        run_classical(D, FEATS, os.path.join(pred_dir, 'predictions_classical.csv'))
        for arch in ARCHS: run_dl(D, FEATS, arch, os.path.join(pred_dir, f'predictions_dl_{arch}.csv'), mk)
    run_classical_tuned(D, FEATS, os.path.join(pred_dir, 'predictions_tuned_classical.csv'), mk, TUNE_LOG)
    for arch in ARCHS:
        run_dl_tuned(D, FEATS, arch, os.path.join(pred_dir, f'predictions_tuned_dl_{arch}.csv'), mk, TUNE_LOG)
    ensemble_tuned(pred_dir)
    print(mk, 'tuned arm done', flush=True)
if TUNE_LOG: pd.DataFrame(TUNE_LOG).to_csv(os.path.join(OUT_ROOT, 'tuning_log.csv'), mode='a', index=False,
                                          header=not os.path.exists(os.path.join(OUT_ROOT, 'tuning_log.csv')))

# %%
TT_all, TC_all = [], []
for mk in TUNED_MARKETS:
    a, b = inference_tuned(mk, os.path.join(OUT_ROOT, mk)); TT_all.append(a); TC_all.append(b)
TT_all, TC_all = pd.concat(TT_all), pd.concat(TC_all)
TT_all.to_csv(os.path.join(OUT_ROOT, 'all_markets_tuned_skill.csv'), index=False)
TC_all.to_csv(os.path.join(OUT_ROOT, 'all_markets_tuned_contrasts.csv'), index=False)
frozen_wide = pd.read_csv(os.path.join(OUT_ROOT, 'all_markets_tableA_skill.csv')).pivot_table(index='method', columns='market', values='bacc_mean')
tuned_wide = TT_all.pivot_table(index='method', columns='market', values='bacc_mean', sort=False)[TUNED_MARKETS]
print('Tuned arm: four-year mean balanced accuracy (%)'); print(tuned_wide.round(2).to_string())
print('\nBest frozen model per market:', frozen_wide.max().round(2).to_dict())
sig = TC_all[(TC_all.p_daily_holm < 0.05) | (TC_all.p_year_holm < 0.05)]
print('\nTuned contrasts significant after Holm:'); print(sig.round(3).to_string() if len(sig) else 'none')
print('\nPredicted Up share (%), tuned:'); print(TT_all.pivot_table(index='method', columns='market', values='up_pred', sort=False)[TUNED_MARKETS].round(1).to_string())
