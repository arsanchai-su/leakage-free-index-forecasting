# %% [markdown]
# # Extended arm: recent architectures and a broader random search (added in revision)
#
# Same point-in-time contract as the frozen and tuned arms: expanding training window from 2012, last pre-test row purged,
# validation = last 250 sessions of each training window with a one-row purge, scalers fitted on fitting rows only, and
# the 2022-2025 test years used once after every choice has been made.
#
# * Architectures: LSTM and GRU (recurrent), TCN (dilated causal convolution), DLinear (decomposition + linear), and a
#   channel-independent PatchTST-style Transformer.
# * Search: N_CONFIGS random configurations per architecture, drawn ONCE from a fixed seed so that every market and fold
#   sees the same candidates (window, width, depth, dropout, learning rate, weight decay, batch size).
# * Selection: per fold, the configuration and epoch with the highest validation balanced accuracy (search seed 42),
#   with early stopping (max 30 epochs, patience 5); the winner is refitted on the full training window with all seeds.
# * Loss: class-weighted BCE (pos_weight = n_down / n_up), threshold 0.5, as in the tuned arm.

# %%
import math, random
EXT_ARCHS = [a for a in os.environ.get('EXT_ARCHS', 'LSTM,GRU,TCN,DLinear,PatchTST').split(',') if a]
ALL_EXT_ARCHS = ['LSTM', 'GRU', 'TCN', 'DLinear', 'PatchTST', 'MLP']
N_CONFIGS = 2 if QUICK else 20
EXT_MAX_EP, EXT_PATIENCE, EXT_VAL_N = (2 if QUICK else 30), (1 if QUICK else 5), 250
EXT_MARKETS = [m for m in os.environ.get('EXT_MARKETS', ','.join(MARKETS)).split(',') if m]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.use_deterministic_algorithms(True, warn_only=True)
else:
    torch.set_num_threads(max(1, os.cpu_count() or 1))
print('extended arm on', DEVICE, '| configs per architecture:', N_CONFIGS)


def search_space(arch, n, seed=20261022):
    """Pre-specified random search space; drawn once, identical for every market and fold."""
    r = random.Random(seed + ALL_EXT_ARCHS.index(arch))
    cfgs = []
    for _ in range(n):
        c = dict(window=r.choice([5, 10, 20, 40, 60]), width=r.choice([16, 32, 64, 128]), layers=r.choice([1, 2]),
                 dropout=r.choice([0.0, 0.1, 0.2, 0.3]), lr=10 ** r.uniform(-4, math.log10(3e-3)),
                 wd=10 ** r.uniform(-6, -3), batch=r.choice([32, 64]))
        if arch == 'PatchTST': c['window'] = r.choice([20, 40, 60])
        cfgs.append(c)
    return cfgs


class RNNx(nn.Module):
    def __init__(s, kind, nf, c):
        super().__init__()
        rnn = nn.LSTM if kind == 'LSTM' else nn.GRU
        s.rnn = rnn(nf, c['width'], num_layers=c['layers'], batch_first=True, dropout=c['dropout'] if c['layers'] > 1 else 0.0)
        s.head = nn.Sequential(nn.Dropout(c['dropout']), nn.Linear(c['width'], 1))
    def forward(s, x): return s.head(s.rnn(x)[0][:, -1])


class TCNBlock(nn.Module):
    def __init__(s, i, o, d, p):
        super().__init__(); s.pad = 2 * d
        s.c1, s.c2 = nn.Conv1d(i, o, 3, dilation=d), nn.Conv1d(o, o, 3, dilation=d)
        s.drop = nn.Dropout(p); s.res = nn.Conv1d(i, o, 1) if i != o else nn.Identity()
    def forward(s, x):
        h = s.drop(torch.relu(s.c1(nn.functional.pad(x, (s.pad, 0)))))
        h = s.drop(torch.relu(s.c2(nn.functional.pad(h, (s.pad, 0)))))
        return torch.relu(h + s.res(x))


class TCN(nn.Module):
    """Dilated causal convolutions (dilations 1, 2, 4[, 8]); the receptive field covers windows up to 60 sessions."""
    def __init__(s, nf, c):
        super().__init__(); w = c['width']; dil = [1, 2, 4] if c['layers'] == 1 else [1, 2, 4, 8]
        s.blocks = nn.Sequential(*[TCNBlock(nf if k == 0 else w, w, d, c['dropout']) for k, d in enumerate(dil)])
        s.head = nn.Linear(w, 1)
    def forward(s, x): return s.head(s.blocks(x.transpose(1, 2))[:, :, -1])


class DLinear(nn.Module):
    """Moving-average trend/remainder decomposition with one linear map over time per component (shared across
    channels), followed by a linear read-out across channels."""
    def __init__(s, nf, c):
        super().__init__(); W = c['window']; s.k = max(3, (W // 4) | 1)
        s.lt, s.ls = nn.Linear(W, 1), nn.Linear(W, 1)
        s.head = nn.Sequential(nn.Dropout(c['dropout']), nn.Linear(nf, 1))
    def forward(s, x):
        z = x.transpose(1, 2)                                      # B, F, W
        pad = (s.k - 1) // 2
        trend = nn.functional.avg_pool1d(nn.functional.pad(z, (pad, pad), mode='replicate'), s.k, stride=1)
        return s.head((s.lt(trend) + s.ls(z - trend)).squeeze(-1))


class PatchTST(nn.Module):
    """Channel-independent patch Transformer: each feature's window is cut into patches of 8 sessions (stride 4),
    embedded, encoded by a Transformer encoder, pooled, and the per-channel outputs are combined linearly."""
    def __init__(s, nf, c):
        super().__init__(); W, d = c['window'], c['width']; s.L, s.S = 8, 4
        s.P = (W - s.L) // s.S + 1
        s.emb = nn.Linear(s.L, d); s.pos = nn.Parameter(torch.zeros(1, s.P, d))
        enc = nn.TransformerEncoderLayer(d, 4, 2 * d, c['dropout'], batch_first=True)
        s.enc = nn.TransformerEncoder(enc, c['layers'], enable_nested_tensor=False)
        s.tok = nn.Linear(d, 1); s.head = nn.Sequential(nn.Dropout(c['dropout']), nn.Linear(nf, 1))
    def forward(s, x):
        B, W, F = x.shape
        z = x.transpose(1, 2).reshape(B * F, W).unfold(1, s.L, s.S)        # B*F, P, L
        h = s.enc(s.emb(z) + s.pos).mean(1)                                  # B*F, d
        return s.head(s.tok(h).reshape(B, F))


class MLP(nn.Module):
    """Feed-forward network on the flattened input window (added at the editor's request)."""
    def __init__(s, nf, c):
        super().__init__(); d = nf * c['window']; layers = []
        for _ in range(c['layers']):
            layers += [nn.Linear(d, c['width']), nn.ReLU(), nn.Dropout(c['dropout'])]; d = c['width']
        s.net = nn.Sequential(nn.Flatten(), *layers, nn.Linear(d, 1))
    def forward(s, x): return s.net(x)


def make_ext(arch, nf, c):
    if arch in ('LSTM', 'GRU'): return RNNx(arch, nf, c)
    return {'TCN': TCN, 'DLinear': DLinear, 'PatchTST': PatchTST, 'MLP': MLP}[arch](nf, c)


def n_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def fit_ext(arch, c, Xtr, ytr, sd, epochs=None, Xval=None, yval=None):
    torch.manual_seed(sd); np.random.seed(sd); random.seed(sd)
    net = make_ext(arch, Xtr.shape[2], c).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), c['lr'], weight_decay=c['wd'])
    pw = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32, device=DEVICE)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    xt = torch.from_numpy(Xtr).to(DEVICE); yt = torch.from_numpy(ytr[:, None].astype(np.float32)).to(DEVICE)
    xv = torch.from_numpy(Xval).to(DEVICE) if Xval is not None else None
    best, best_ep, bad = -1.0, 1, 0
    for ep in range(1, (epochs or EXT_MAX_EP) + 1):
        net.train()
        for b in range(0, len(xt), c['batch']):            # unshuffled mini-batches, as in the other arms
            opt.zero_grad(); loss = lossf(net(xt[b:b + c['batch']]), yt[b:b + c['batch']]); loss.backward(); opt.step()
        if epochs is None:
            net.eval()
            with torch.no_grad(): pv = (net(xv) > 0).cpu().numpy().ravel().astype(int)
            v = bacc(yval, pv); v = -1.0 if np.isnan(v) else v
            if v > best + 1e-9: best, best_ep, bad = v, ep, 0
            else:
                bad += 1
                if bad >= EXT_PATIENCE: break
    return net, best_ep, best


def predict_ext(net, X):
    net.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(X), 512): out.append(net(torch.from_numpy(X[b:b + 512]).to(DEVICE)).cpu().numpy().ravel())
    return 1 / (1 + np.exp(-np.concatenate(out)))


def run_extended(D, FEATS, arch, path, mk, log_path):
    if os.path.exists(path): return
    X_all, y, out, logs = D[FEATS].values, D.y.values, [], []
    cfgs = search_space(arch, N_CONFIGS)
    for year, s, e in FOLDS:
        te_idx = np.where((D.Date >= s) & (D.Date <= e))[0]
        if len(te_idx) == 0: continue
        base_idx = np.where(D.Date < s)[0][:-1]                       # outer purge
        res = []
        for ci, c in enumerate(cfgs):
            W = c['window']; tr_idx = base_idx[base_idx >= W - 1]
            fit_idx, val_idx = tr_idx[:-(EXT_VAL_N + 1)], tr_idx[-EXT_VAL_N:]      # inner purge
            sf = StandardScaler().fit(X_all[fit_idx]); Xf = np.clip(sf.transform(X_all), -10, 10)
            t0 = time.time()
            _, ep, vb = fit_ext(arch, c, seqs(Xf, fit_idx, W), y[fit_idx], SEEDS[0], None, seqs(Xf, val_idx, W), y[val_idx])
            res.append((vb, -ci, ci, ep))
            print(f'{mk:7s} {arch:9s} {year} cfg {ci:2d} W={W:2d} width={c["width"]:3d} L={c["layers"]} '
                  f'val {vb:.3f} ep {ep:2d} {time.time() - t0:5.1f}s', flush=True)
        vb, _, ci, ep = max(res)                                        # ties -> earlier configuration
        c = cfgs[ci]; W = c['window']; tr_idx = base_idx[base_idx >= W - 1]
        st = StandardScaler().fit(X_all[tr_idx]); Xs = np.clip(st.transform(X_all), -10, 10)
        te = D.iloc[te_idx]
        for sd in SEEDS:
            net, _, _ = fit_ext(arch, c, seqs(Xs, tr_idx, W), y[tr_idx], sd, epochs=ep)
            p = predict_ext(net, seqs(Xs, te_idx, W))
            out.append(frame(f'{arch} (extended)', 'Extended', sd, year, te, p > 0.5, p, arch=arch))
        logs.append(dict(market=mk, year=year, arch=arch, config=ci, epochs=ep, val_bacc=vb, params=n_params(net),
                         n_train=len(tr_idx), **c))
    pd.concat(out).to_csv(path, index=False)
    pd.DataFrame(logs).to_csv(log_path, mode='a', index=False, header=not os.path.exists(log_path))

# %%
def ensure_baselines(D, FEATS, pred_dir, mk):
    """The extended arm is compared with frozen logistic regression, persistence, and the tuned random forest.
    If the earlier arms were run in the same results folder their predictions are reused; otherwise they are refitted."""
    if not os.path.exists(os.path.join(pred_dir, 'predictions_classical.csv')):
        run_classical(D, FEATS, os.path.join(pred_dir, 'predictions_classical.csv'))
    if not os.path.exists(os.path.join(pred_dir, 'predictions_tuned_classical.csv')):
        run_classical_tuned(D, FEATS, os.path.join(pred_dir, 'predictions_tuned_classical.csv'), mk, [])


t_ext = time.time()
for mk in EXT_MARKETS:
    out_dir = os.path.join(OUT_ROOT, mk); pred_dir = os.path.join(out_dir, 'predictions'); os.makedirs(pred_dir, exist_ok=True)
    A = build_aligned(mk)
    has_vol = A['Volume_D'].notna().all() and (A['Volume_D'] > 0).all()
    use_vol = has_vol if USE_VOLUME == 'auto' else False
    D, FEATS = build_features(A, use_vol)
    ensure_baselines(D, FEATS, pred_dir, mk)
    for arch in EXT_ARCHS:
        run_extended(D, FEATS, arch, os.path.join(pred_dir, f'predictions_ext_{arch}.csv'), mk, os.path.join(OUT_ROOT, 'extended_search_log.csv'))
    print(f'{mk} extended arm done, elapsed {(time.time() - t_ext) / 60:.1f} min', flush=True)

# %%
Z90 = stats.norm.ppf(0.95)


def inference_extended(mk, out_dir):
    rng = np.random.default_rng(20260926)
    pred_dir = os.path.join(out_dir, 'predictions')
    files = [os.path.join(pred_dir, f) for f in ['predictions_classical.csv', 'predictions_tuned_classical.csv']]
    files += [os.path.join(pred_dir, f'predictions_ext_{a}.csv') for a in EXT_ARCHS]
    allp = pd.concat([pd.read_csv(f, parse_dates=['Date']) for f in files], ignore_index=True)
    allp = allp[~allp.zero].copy(); allp['q'] = allp.Date.dt.year.astype(str) + 'Q' + allp.Date.dt.quarter.astype(str)
    ext = [f'{a} (extended)' for a in EXT_ARCHS]

    def ens(m):
        g = allp[allp.method == m]
        s = g.groupby('Date').agg(y=('y', 'first'), score=('score', 'mean'), year=('year', 'first'), q=('q', 'first'))
        fam = g.family.iloc[0]
        s['pred'] = ((s.score > 0) if fam == 'Naive' else (s.score > 0.5)).astype(int)
        return s.sort_index()

    names = ext + ['Logistic-L2', 'Persistence', 'Random forest (tuned)']
    E = {m: ens(m) for m in names}
    n = len(E[ext[0]]); k = int(np.ceil(n / BLOCK)); starts = rng.integers(0, n, size=(B, k))
    IDX = ((starts[:, :, None] + np.arange(BLOCK)[None, None, :]) % n).reshape(B, -1)[:, :n]
    rows = []
    for m in ext:
        g = allp[allp.method == m]
        per = g.groupby(['year', 'seed']).apply(lambda t: bacc(t.y.values, t.pred.values)).groupby('year').mean()
        y, p = E[m].y.values, E[m].pred.values; bb = boot_bacc(y, p, IDX); s_, ppt = pt_test(y, p)
        rows.append(dict(market=mk, method=m, **{f'bacc_{yr}': 100 * v for yr, v in per.items()}, bacc_mean=100 * per.mean(),
                         bacc_pooled=100 * bacc(y, p), ci_lo=100 * np.nanpercentile(bb, 2.5), ci_hi=100 * np.nanpercentile(bb, 97.5),
                         acc=100 * (y == p).mean(), up_pred=100 * p.mean(), pt_p=ppt, n=len(y)))
    TS = pd.DataFrame(rows); TS['pt_holm'] = holm(TS.pt_p.fillna(1).values)

    def ub(m, unit):
        g = allp[allp.method == m]
        return g.groupby([unit, 'seed']).apply(lambda t: bacc(t.y.values, t.pred.values)).groupby(unit).mean()

    def con(a, b, fam):
        dy = (ub(a, 'year') - ub(b, 'year')).values; dq = (ub(a, 'q') - ub(b, 'q')).values
        y = E[a].y.values; dd = boot_bacc(y, E[a].pred.values, IDX) - boot_bacc(y, E[b].pred.values, IDX)
        obs = bacc(y, E[a].pred.values) - bacc(y, E[b].pred.values); se = np.nanstd(dd)
        tost = max(1 - stats.norm.cdf((obs + 0.02) / se), stats.norm.cdf((obs - 0.02) / se)) if se > 0 else 1.0
        return dict(market=mk, family=fam, A=a, B=b, delta_year_mean=100 * dy.mean(), p_year=signflip_p(dy),
                    p_quarter=signflip_p(dq), delta_daily=100 * obs, se_daily=100 * se,
                    p_daily=float(2 * (1 - stats.norm.cdf(abs(obs) / se))) if se > 0 else 1.0, mde80=100 * 2.8 * se,
                    min_equiv_margin=100 * (abs(obs) + Z90 * se), p_tost_2pp=tost)

    R = [con(m, 'Logistic-L2', 'E1 extended vs logistic') for m in ext]
    R += [con(m, 'Persistence', 'E2 extended vs persistence') for m in ext]
    R += [con(m, 'Random forest (tuned)', 'E3 extended vs tuned random forest') for m in ext]
    TC = pd.DataFrame(R)
    for c in ['p_year', 'p_quarter', 'p_daily', 'p_tost_2pp']:
        TC[c + '_holm'] = TC.groupby('family')[c].transform(lambda s: holm(s.values))
    TS.to_csv(os.path.join(out_dir, 'tableE_extended_skill.csv'), index=False)
    TC.to_csv(os.path.join(out_dir, 'tableE_extended_contrasts.csv'), index=False)
    return TS, TC


ES, EC = [], []
for mk in EXT_MARKETS:
    a, b = inference_extended(mk, os.path.join(OUT_ROOT, mk)); ES.append(a); EC.append(b)
ES, EC = pd.concat(ES), pd.concat(EC)
ES['pt_holm_all'] = holm(ES.pt_p.fillna(1).values)
ES.to_csv(os.path.join(OUT_ROOT, 'all_markets_extended_skill.csv'), index=False)
EC.to_csv(os.path.join(OUT_ROOT, 'all_markets_extended_contrasts.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print('Extended arm: four-year mean balanced accuracy (%)')
print(ES.pivot_table(index='method', columns='market', values='bacc_mean', sort=False)[EXT_MARKETS].round(2).to_string())
print('\nPooled seed-ensemble balanced accuracy (%) [95% CI]')
ES['pooled_ci'] = ES.apply(lambda r: f'{r.bacc_pooled:.2f} [{r.ci_lo:.1f}, {r.ci_hi:.1f}]', axis=1)
print(ES.pivot_table(index='method', columns='market', values='pooled_ci', aggfunc='first', sort=False)[EXT_MARKETS].to_string())
print('\nPredicted up share (%)'); print(ES.pivot_table(index='method', columns='market', values='up_pred', sort=False)[EXT_MARKETS].round(1).to_string())
print('\nPesaran-Timmermann, Holm within market / across all:'); print(ES[['market', 'method', 'pt_p', 'pt_holm', 'pt_holm_all']].round(4).to_string())
sig = EC[(EC.p_daily_holm < 0.05) | (EC.p_quarter_holm < 0.05)]
print('\nContrasts significant after Holm:'); print(sig[['market', 'family', 'A', 'delta_daily', 'p_daily_holm', 'p_quarter_holm']].round(3).to_string() if len(sig) else 'none')
print('\nEquivalence within 2 pp after Holm:', int((EC.p_tost_2pp_holm < 0.05).sum()), 'of', len(EC),
      f'| smallest margin supported: {EC.min_equiv_margin.min():.1f}-{EC.min_equiv_margin.max():.1f} pp')
LOG = pd.read_csv(os.path.join(OUT_ROOT, 'extended_search_log.csv'))
print('\nSelected configurations: median validation BAcc, epochs, window, params by architecture')
print(LOG.groupby('arch').agg(val_bacc=('val_bacc', 'median'), epochs=('epochs', 'median'), window=('window', 'median'),
                               params_min=('params', 'min'), params_max=('params', 'max')).round(3).to_string())
