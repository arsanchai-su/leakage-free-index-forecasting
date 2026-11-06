# %% [markdown]
# # Benchmark calibration across four stock indices: SET50, SET100, S&P 500, CSI 300
#
# One notebook re-runs the frozen point-in-time protocol of the SET50 study (Sections 3.9 and 4.7, Tables 8-9, Figure 10) on
# four indices with freshly downloaded Investing.com data (retrieved 2026-10-02):
#
# | Market | Index | Volume features |
# |---|---|---|
# | SET50 | Stock Exchange of Thailand, top 50 | yes (85 features) |
# | SET100 | Stock Exchange of Thailand, top 100 | yes (85 features) |
# | SPX | S&P 500 | no, Investing.com has no SPX volume (75 features) |
# | CSI300 | CSI 300, Shanghai and Shenzhen | yes (85 features) |
#
# Protocol (identical for every market, nothing tuned on test years): outer test years 2022-2025 with an expanding
# training window and the last pre-test row purged; weekly and monthly inputs use only the previous completed bar;
# scalers fitted on training rows; naive rules, classical learners, and five deep networks (LSTM, CNN, LSTM-CNN,
# LSTM-Attention, LSTM-CNN-Attention) under level-regression and direct-BCE objectives; seeds 42, 123, 456, 789, 2025;
# balanced accuracy as the primary endpoint; Holm-adjusted paired contrasts and minimum detectable effects.
#
# **Run**: `Runtime > Run all`. CPU is enough. About 25-40 minutes per market. Every finished model is saved, so after a
# disconnect just run all again and finished parts are skipped. If the data folder is in Google Drive
# (`MyDrive/Run_4Markets`), results are written there too and survive a runtime reset.

# %%
# ---------------- configuration ----------------
MARKETS = ['SET50', 'SET100', 'SPX', 'CSI300']
USE_VOLUME = 'auto'      # 'auto': volume features when the market has volume; False: same 75 price-only features for all markets
INCLUDE_2026 = False     # True adds the partial 2026 year (Jan-Sep) as an extra outer fold
QUICK = False            # True = smoke test (1 seed, 2 epochs, 2 architectures)
DRIVE_DIR = '/content/drive/MyDrive/Run_4Markets'   # folder that contains data/ (used if present)
LOCAL_DIR = '/content/Run_4Markets'                 # used when Drive is not mounted

FOLDS = [(y, f'{y}-01-01', f'{y}-12-31') for y in (2022, 2023, 2024, 2025)]
if INCLUDE_2026: FOLDS.append((2026, '2026-01-01', '2026-12-31'))
SEEDS = [42] if QUICK else [42, 123, 456, 789, 2025]
EPOCHS = 2 if QUICK else 20
WIN = {'LSTM': 5, 'CNN': 20, 'LSTM-CNN': 20, 'LSTM-Attention': 10, 'LSTM-CNN-Attention': 20}
ARCHS = ['LSTM', 'CNN'] if QUICK else list(WIN)
TAG = ('quick_' if QUICK else '') + ('pricevol' if USE_VOLUME == 'auto' else 'priceonly') + ('_2026' if INCLUDE_2026 else '')

# %%
# ---------------- locate data: Google Drive folder, or upload Run_4Markets_data.zip ----------------
import os, shutil, zipfile, importlib.util, subprocess, sys
for pkg in ['lightgbm']:
    if importlib.util.find_spec(pkg) is None:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=True)
ROOT = None
IN_COLAB = importlib.util.find_spec('google.colab') is not None
if IN_COLAB:
    from google.colab import drive
    try:
        drive.mount('/content/drive')
        os.makedirs(DRIVE_DIR, exist_ok=True)   # results are always saved in Google Drive when it is mounted
        ROOT = DRIVE_DIR
    except Exception as e:
        print('WARNING: Google Drive could not be mounted; results will be lost if the runtime resets:', e)
else:
    print('Drive not used: not running in Colab')
if ROOT is None:
    ROOT = LOCAL_DIR
    os.makedirs(ROOT, exist_ok=True)
if not (os.path.isdir(os.path.join(ROOT, 'data')) or os.path.isdir(os.path.join(ROOT, 'Run_4Markets', 'data'))):
    from google.colab import files
    print('data/ not found in', ROOT, '- upload Run_4Markets_data.zip (it is extracted into this folder once)')
    for name in files.upload():
        with zipfile.ZipFile(name) as z: z.extractall(ROOT)
DATA_DIR = os.path.join(ROOT, 'data')
if not os.path.isdir(DATA_DIR) and os.path.isdir(os.path.join(ROOT, 'Run_4Markets', 'data')):
    DATA_DIR = os.path.join(ROOT, 'Run_4Markets', 'data')
OUT_ROOT = os.path.join(ROOT, 'results', TAG)
print('data:', DATA_DIR, sorted(os.listdir(DATA_DIR)))
print('results:', OUT_ROOT)

# %%
# ---------------- verify files against the manifest ----------------
import hashlib, json
man = json.load(open(os.path.join(DATA_DIR, 'manifest.json')))
for f in man['files']:
    p = os.path.join(DATA_DIR, f['name'])
    ok = os.path.exists(p) and hashlib.sha256(open(p, 'rb').read()).hexdigest() == f['sha256']
    print(('OK      ' if ok else 'CHANGED ') + f['name'])

# %%
# ---------------- data alignment and features ----------------
import numpy as np, pandas as pd
COLS = ['Close', 'Open', 'High', 'Low', 'Volume', 'Change_pct']
LAST_TEST = max(e for _, _, e in FOLDS)


def load(name):
    return pd.read_csv(os.path.join(DATA_DIR, name), parse_dates=['Date']).sort_values('Date').reset_index(drop=True)


def build_aligned(mk):
    """Daily rows + previous completed week (weeks start Sunday) + previous completed month; target = next session's close."""
    d, w, m = load(f'{mk}_days.csv'), load(f'{mk}_weeks.csv'), load(f'{mk}_months.csv')
    d['Target_Next_Close'] = d.Close.shift(-1)
    d = d[d.Date <= LAST_TEST]
    D = d.rename(columns={c: c + '_D' for c in COLS})
    D['wkey'] = D.Date - pd.to_timedelta((D.Date.dt.dayofweek + 1) % 7, unit='D') - pd.Timedelta(days=7)
    W = w[['Date'] + COLS].rename(columns={c: c + '_W' for c in COLS}).rename(columns={'Date': 'wdate'})
    D = pd.merge_asof(D.sort_values('wkey'), W.sort_values('wdate'), left_on='wkey', right_on='wdate').sort_values('Date')
    D['mkey'] = D.Date.dt.to_period('M').dt.start_time - pd.offsets.MonthBegin(1)
    M = m[['Date'] + COLS].rename(columns={c: c + '_M' for c in COLS}).rename(columns={'Date': 'mdate'})
    D = pd.merge_asof(D.sort_values('mkey'), M.sort_values('mdate'), left_on='mkey', right_on='mdate').sort_values('Date')
    keep = ['Date'] + [c + s for s in ['_D', '_W', '_M'] for c in COLS] + ['Target_Next_Close']
    D = D.dropna(subset=[c for c in keep if not c.startswith('Volume')])
    assert (D.wdate + pd.Timedelta(days=7) <= D.Date).all() and (D.mdate < D.Date.dt.to_period('M').dt.start_time).all()
    return D[keep].reset_index(drop=True)


def rsi(c, n=14):
    d = c.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    au = up.ewm(alpha=1/n, adjust=False).mean(); ad = dn.ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1 + au/ad.replace(0, np.nan))


def adx(h, l, c, n=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    upm = h.diff(); dnm = -l.diff()
    pdm = np.where((upm > dnm) & (upm > 0), upm, 0.0); ndm = np.where((dnm > upm) & (dnm > 0), dnm, 0.0)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100*pd.Series(pdm, c.index).ewm(alpha=1/n, adjust=False).mean()/atr
    ndi = 100*pd.Series(ndm, c.index).ewm(alpha=1/n, adjust=False).mean()/atr
    dx = 100*(pdi-ndi).abs()/(pdi+ndi)
    return dx.ewm(alpha=1/n, adjust=False).mean(), pdi, ndi, atr


def build_features(a, use_vol):
    a = a.sort_values('Date').reset_index(drop=True)
    c, o, h, l = a.Close_D, a.Open_D, a.High_D, a.Low_D
    lv = np.log(a.Volume_D) if use_vol else None
    F = pd.DataFrame(index=a.index)
    for col in ['Close_D', 'Open_D', 'High_D', 'Low_D']: F[col] = a[col]
    if use_vol: F['logVol_D'] = lv
    for k in range(1, 6):
        F[f'close_lag{k}'] = c.shift(k)
        if use_vol: F[f'logvol_lag{k}'] = lv.shift(k)
    r1 = c.pct_change()
    for k in [1, 3, 5, 10, 20, 60]: F[f'ret_{k}'] = c.pct_change(k)
    for k in [5, 10, 20, 60]:
        ma = c.rolling(k).mean(); F[f'ma_{k}'] = ma; F[f'c_ma_{k}'] = c/ma - 1
        F[f'vol_{k}'] = r1.rolling(k).std()
        F[f'mom_{k}'] = c - c.shift(k); F[f'roc_{k}'] = 100*(c/c.shift(k) - 1)
        if use_vol: F[f'vr_{k}'] = lv - lv.rolling(k).mean()
    F['ma5_ma20'] = F.ma_5/F.ma_20 - 1; F['ma20_ma60'] = F.ma_20/F.ma_60 - 1
    rng_ = (h - l).replace(0, np.nan)
    F['body'] = (c - o)/o; F['range'] = (h - l)/c
    F['upper_sh'] = (h - np.maximum(o, c))/rng_; F['lower_sh'] = (np.minimum(o, c) - l)/rng_
    F['close_pos'] = (c - l)/rng_; F['gap'] = o/c.shift() - 1
    for k in range(1, 6):
        F[f'dir_lag{k}'] = (c.shift(k-1) > c.shift(k)).astype(float); F[f'ret_lag{k}'] = r1.shift(k-1)
    ll14, hh14 = l.rolling(14).min(), h.rolling(14).max()
    F['stoch_k'] = 100*(c - ll14)/(hh14 - ll14); F['stoch_d'] = F.stoch_k.rolling(3).mean()
    F['rsi14'] = rsi(c, 14); F['rsi6'] = rsi(c, 6)
    e12, e26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    F['macd'] = e12 - e26; F['macd_sig'] = F.macd.ewm(span=9, adjust=False).mean(); F['macd_hist'] = F.macd - F.macd_sig
    F['willr'] = -100*(hh14 - c)/(hh14 - ll14)
    tp = (h + l + c)/3
    F['cci20'] = (tp - tp.rolling(20).mean())/(0.015*tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True))
    F['adx14'], F['pdi14'], F['ndi14'], atr = adx(h, l, c); F['atr14_rel'] = atr/c
    for p in ['W', 'M']:
        F[f'chg_{p}'] = a[f'Change_pct_{p}']; F[f'c_vs_close_{p}'] = c/a[f'Close_{p}'] - 1
        F[f'range_{p}'] = (a[f'High_{p}'] - a[f'Low_{p}'])/a[f'Close_{p}']
        F[f'body_{p}'] = (a[f'Close_{p}'] - a[f'Open_{p}'])/a[f'Open_{p}']
    F['chg_D'] = a.Change_pct_D
    out = pd.concat([pd.DataFrame({'Date': a.Date, 'C': c, 'C_next': a.Target_Next_Close}), F], axis=1)
    out['ret_next'] = out.C_next/out.C - 1
    out['y'] = (out.C_next > out.C).astype(int); out['zero'] = out.C_next == out.C
    out = out.replace([np.inf, -np.inf], np.nan).iloc[60:].reset_index(drop=True)
    feats = list(F.columns)
    bad = out[feats].isna().sum(); assert bad.sum() == 0, bad[bad > 0]
    return out, feats

# %%
# ---------------- forecasters ----------------
import time, warnings, torch, torch.nn as nn, lightgbm as lgb
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
warnings.filterwarnings('ignore')
torch.set_num_threads(1); torch.use_deterministic_algorithms(True)


def frame(method, family, seed, year, te, pred, score, arch=None):
    f = pd.DataFrame({'method': method, 'family': family, 'seed': seed, 'year': year, 'Date': te.Date.values, 'y': te.y.values,
                      'zero': te.zero.values, 'pred': np.asarray(pred).astype(int), 'score': np.asarray(score, float)})
    if arch: f.insert(2, 'arch', arch)
    return f


def run_classical(D, FEATS, path):
    if os.path.exists(path): return
    rows = []
    for year, s, e in FOLDS:
        tr = D[D.Date < s].iloc[:-1]; te = D[(D.Date >= s) & (D.Date <= e)]
        if te.empty: continue
        Xtr, Xte = tr[FEATS].values, te[FEATS].values
        sc = StandardScaler().fit(Xtr); Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
        ytr = tr.y.values; maj = int(ytr.mean() >= 0.5)
        rows += [frame('Train-majority', 'Naive', 0, year, te, np.full(len(te), maj), np.full(len(te), ytr.mean())),
                 frame('Persistence', 'Naive', 0, year, te, te.dir_lag1.values, te.ret_1.values),
                 frame('Reversal', 'Naive', 0, year, te, 1 - te.dir_lag1.values, -te.ret_1.values),
                 frame('MA20-trend', 'Naive', 0, year, te, (te.c_ma_20 > 0).astype(int).values, te.c_ma_20.values)]
        p = LogisticRegression(C=1.0, max_iter=5000).fit(Ztr, ytr).predict_proba(Zte)[:, 1]
        rows.append(frame('Logistic-L2', 'Classical', 0, year, te, p > 0.5, p))
        r = Ridge(alpha=1.0).fit(Ztr, tr.ret_next.values).predict(Zte)
        rows.append(frame('Ridge-return', 'Classical', 0, year, te, r > 0, r))
        for sd in SEEDS:
            p = RandomForestClassifier(n_estimators=500, min_samples_leaf=20, max_features='sqrt', n_jobs=-1,
                                       random_state=sd).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
            rows.append(frame('Random forest', 'Classical', sd, year, te, p > 0.5, p))
            p = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, random_state=sd, verbose=-1).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
            rows.append(frame('LightGBM', 'Classical', sd, year, te, p > 0.5, p))
    pd.concat(rows).to_csv(path, index=False)


class CausalConv(nn.Module):
    def __init__(s, i, o, k=3):
        super().__init__(); s.k = k; s.c = nn.Conv1d(i, o, k)
    def forward(s, x):
        x = nn.functional.pad(x.transpose(1, 2), (s.k - 1, 0)); return torch.relu(s.c(x)).transpose(1, 2)


class CausalAtt(nn.Module):
    def __init__(s, d):
        super().__init__(); s.m = nn.MultiheadAttention(d, 2, batch_first=True, kdim=d, vdim=d)
    def forward(s, x):
        T = x.shape[1]; mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        return s.m(x, x, x, attn_mask=mask, need_weights=False)[0]


class Net(nn.Module):
    def __init__(s, arch, nf):
        super().__init__(); s.arch = arch
        if 'LSTM' in arch: s.lstm = nn.LSTM(nf, 16, batch_first=True)
        dim = 16 if 'LSTM' in arch else nf
        if 'CNN' in arch: s.conv = CausalConv(dim, 32); dim = 32
        if 'Attention' in arch: s.att = CausalAtt(dim)
        s.head = nn.Sequential(nn.Linear(dim, 8), nn.ReLU(), nn.Linear(8, 1))
    def forward(s, x):
        if s.arch == 'LSTM': return s.head(s.lstm(x)[0][:, -1])
        if 'LSTM' in s.arch: x = s.lstm(x)[0]
        if 'CNN' in s.arch: x = s.conv(x)
        if 'Attention' in s.arch: x = s.att(x)
        return s.head(x.mean(1))


def seqs(X, idx, W):
    return np.stack([X[i - W + 1:i + 1] for i in idx]).astype(np.float32)


def run_dl(D, FEATS, arch, path, mk):
    if os.path.exists(path): return
    X_all, W, out = D[FEATS].values, WIN[arch], []
    for year, s, e in FOLDS:
        tr_idx = np.where(D.Date < s)[0][:-1]; tr_idx = tr_idx[tr_idx >= W - 1]
        te_idx = np.where((D.Date >= s) & (D.Date <= e))[0]
        if len(te_idx) == 0: continue
        sx = MinMaxScaler().fit(X_all[tr_idx]); Xs = sx.transform(X_all)
        sy = MinMaxScaler().fit(D.C_next.values[tr_idx, None])
        Xtr, Xte = seqs(Xs, tr_idx, W), seqs(Xs, te_idx, W)
        targets = {'R': sy.transform(D.C_next.values[tr_idx, None]).astype(np.float32), 'C': D.y.values[tr_idx, None].astype(np.float32)}
        te = D.iloc[te_idx]
        for obj in ['R', 'C']:
            for sd in SEEDS:
                t0 = time.time(); torch.manual_seed(sd); np.random.seed(sd)
                net = Net(arch, X_all.shape[1]); opt = torch.optim.Adam(net.parameters(), 1e-3)
                lossf = nn.MSELoss() if obj == 'R' else nn.BCEWithLogitsLoss()
                xt, yt = torch.from_numpy(Xtr), torch.from_numpy(targets[obj])
                net.train()
                for ep in range(EPOCHS):
                    for b in range(0, len(xt), 32):
                        opt.zero_grad(); loss = lossf(net(xt[b:b+32]), yt[b:b+32]); loss.backward(); opt.step()
                net.eval()
                with torch.no_grad(): o = net(torch.from_numpy(Xte)).numpy().ravel()
                if obj == 'R': score = sy.inverse_transform(o[:, None]).ravel() - te.C.values; pred = score > 0
                else: score = 1/(1 + np.exp(-o)); pred = score > 0.5
                out.append(frame(f'{arch} ({"level-regression" if obj == "R" else "direct-BCE"})', 'DL-' + obj, sd, year, te, pred, score, arch=arch))
                print(f'{mk:7s} {arch:20s} {year} {obj} seed {sd:4d} {time.time()-t0:5.1f}s', flush=True)
    pd.concat(out).to_csv(path, index=False)

# %%
# ---------------- inference ----------------
import glob, itertools
from scipy import stats
B, BLOCK = 5000, 10


def bacc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if y.min() == y.max(): return np.nan
    return 0.5*((p[y == 1] == 1).mean() + (p[y == 0] == 0).mean())


def mcc(y, p):
    tp = ((y == 1) & (p == 1)).sum(); tn = ((y == 0) & (p == 0)).sum(); fp = ((y == 0) & (p == 1)).sum(); fn = ((y == 1) & (p == 0)).sum()
    den = np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    return (tp*tn - fp*fn)/den if den > 0 else 0.0


def pt_test(y, p):
    n = len(y); py, pp = y.mean(), p.mean(); ph = (y == p).mean()
    pstar = py*pp + (1-py)*(1-pp); v1 = pstar*(1-pstar)/n
    v2 = (2*py-1)**2*pp*(1-pp)/n + (2*pp-1)**2*py*(1-py)/n + 4*py*pp*(1-py)*(1-pp)/n**2
    if v1 - v2 <= 0: return np.nan, np.nan
    s = (ph - pstar)/np.sqrt(v1 - v2); return s, 1 - stats.norm.cdf(s)


def boot_bacc(y, p, idx):
    Y, P = y[idx], p[idx]
    return 0.5*(((Y == 1) & (P == 1)).sum(1)/(Y == 1).sum(1) + ((Y == 0) & (P == 0)).sum(1)/(Y == 0).sum(1))


def signflip_p(effects):
    e = np.asarray(effects, float); e = e[~np.isnan(e)]; k = len(e); obs = abs(e.mean())
    signs = np.array(list(itertools.product([-1, 1], repeat=k)))
    return float(np.mean(np.abs((signs*e).mean(1)) >= obs - 1e-12))


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r)*p[i]); adj[i] = min(1, run)
    return adj


def inference(mk, out_dir, n_feats, use_vol):
    rng = np.random.default_rng(20260926)
    pred_dir = os.path.join(out_dir, 'predictions')
    allp = pd.concat([pd.read_csv(f, parse_dates=['Date']) for f in sorted(glob.glob(os.path.join(pred_dir, 'predictions_*.csv')))], ignore_index=True)
    allp = allp[~allp.zero].copy(); allp['q'] = allp.Date.dt.year.astype(str) + 'Q' + allp.Date.dt.quarter.astype(str)
    order = ['Train-majority', 'Persistence', 'Reversal', 'MA20-trend', 'Logistic-L2', 'Ridge-return', 'Random forest', 'LightGBM']
    order += [f'{a} (level-regression)' for a in ARCHS] + [f'{a} (direct-BCE)' for a in ARCHS]

    def ensemble(m):
        g = allp[allp.method == m]
        s = g.groupby('Date').agg(y=('y', 'first'), score=('score', 'mean'), year=('year', 'first'), q=('q', 'first'))
        fam = g.family.iloc[0]
        if m == 'Train-majority': s['pred'] = g.groupby('Date').pred.first()
        elif fam == 'Naive' or m == 'Ridge-return' or fam == 'DL-R': s['pred'] = (s.score > 0).astype(int)
        else: s['pred'] = (s.score > 0.5).astype(int)
        return s.sort_index()

    ENS = {m: ensemble(m) for m in order}
    n = len(next(iter(ENS.values())))
    k = int(np.ceil(n/BLOCK)); starts = rng.integers(0, n, size=(B, k))
    IDX = ((starts[:, :, None] + np.arange(BLOCK)[None, None, :]) % n).reshape(B, -1)[:, :n]
    rows = []
    for m in order:
        g = allp[allp.method == m]
        per = g.groupby(['year', 'seed']).apply(lambda t: bacc(t.y.values, t.pred.values)).groupby('year').mean()
        e = ENS[m]; y, p = e.y.values, e.pred.values
        bb = boot_bacc(y, p, IDX); s_, ppt = pt_test(y, p)
        rows.append(dict(market=mk, method=m, family=g.family.iloc[0], **{f'bacc_{yr}': 100*v for yr, v in per.items()},
                         bacc_mean=100*per.mean(), bacc_pooled=100*bacc(y, p), ci_lo=100*np.nanpercentile(bb, 2.5),
                         ci_hi=100*np.nanpercentile(bb, 97.5), p_boot_gt50=float(np.mean(bb <= 0.5)), acc=100*(y == p).mean(),
                         mcc=mcc(y, p), up_pred=100*p.mean(), pt_stat=s_, pt_p=ppt, n=len(y)))
    TA = pd.DataFrame(rows); TA['pt_holm'] = holm(TA.pt_p.fillna(1).values)
    TA.to_csv(os.path.join(out_dir, 'tableA_skill.csv'), index=False)

    def unit_bacc(m, unit):
        g = allp[allp.method == m]
        return g.groupby([unit, 'seed']).apply(lambda t: bacc(t.y.values, t.pred.values)).groupby(unit).mean()

    def contrast(a, b, fam):
        dy = (unit_bacc(a, 'year') - unit_bacc(b, 'year')).values; dq = (unit_bacc(a, 'q') - unit_bacc(b, 'q')).values
        ea, eb = ENS[a], ENS[b]; y = ea.y.values
        dd = boot_bacc(y, ea.pred.values, IDX) - boot_bacc(y, eb.pred.values, IDX)
        obs = bacc(y, ea.pred.values) - bacc(y, eb.pred.values); se = np.nanstd(dd)
        return dict(market=mk, family=fam, A=a, B=b, delta_year_mean=100*dy.mean(), p_year=signflip_p(dy), n_q=len(dq),
                    delta_q_mean=100*dq.mean(), p_quarter=signflip_p(dq), delta_daily=100*obs, se_daily=100*se,
                    p_daily=float(2*(1 - stats.norm.cdf(abs(obs)/se))), ci_lo=100*(obs - 1.96*se), ci_hi=100*(obs + 1.96*se),
                    mde80=100*2.8*se, discordance=float((ea.pred.values != eb.pred.values).mean()))

    rowsB = [contrast(f'{a} (direct-BCE)', f'{a} (level-regression)', 'F1 objective') for a in ARCHS]
    rowsB += [contrast(f'{a} (level-regression)', 'Logistic-L2', 'F2 DL vs logistic') for a in ARCHS]
    rowsB += [contrast(f'{a} (level-regression)', 'Persistence', 'F3 DL vs persistence') for a in ARCHS]
    TB = pd.DataFrame(rowsB)
    for col in ['p_year', 'p_quarter', 'p_daily']:
        TB[col + '_holm'] = TB.groupby('family')[col].transform(lambda s: holm(s.values))
    TB.to_csv(os.path.join(out_dir, 'tableB_contrasts.csv'), index=False)
    TC = pd.DataFrame([dict(market=mk, discordance=d_, n=n, mde80_iid_pp=100*2.8*np.sqrt(d_/n)) for d_ in [0.10, 0.20, 0.30, 0.40]])
    TC.to_csv(os.path.join(out_dir, 'tableC_power.csv'), index=False)
    e0 = ENS['Logistic-L2']
    misc = dict(market=mk, volume_features=bool(use_vol), n_features=n_feats, quick=QUICK, n=int(n), up_share=float(e0.y.mean()),
                up_share_by_year={int(k_): float(v) for k_, v in e0.groupby('year').y.mean().items()},
                sessions_by_year={int(k_): int(v) for k_, v in e0.groupby('year').size().items()},
                versions=dict(numpy=np.__version__, pandas=pd.__version__, torch=torch.__version__, lightgbm=lgb.__version__,
                              sklearn=__import__('sklearn').__version__))
    json.dump(misc, open(os.path.join(out_dir, 'misc.json'), 'w'), indent=1)
    return TA, TB, misc

# %%
# ---------------- run all markets ----------------
SUMMARY = {}
t_all = time.time()
for mk in MARKETS:
    out_dir = os.path.join(OUT_ROOT, mk); pred_dir = os.path.join(out_dir, 'predictions'); os.makedirs(pred_dir, exist_ok=True)
    A = build_aligned(mk)
    has_vol = A['Volume_D'].notna().all() and (A['Volume_D'] > 0).all()
    use_vol = has_vol if USE_VOLUME == 'auto' else False
    D, FEATS = build_features(A, use_vol)
    print(f'\n=== {mk}: {len(D)} rows {D.Date.min().date()} -> {D.Date.max().date()} | {len(FEATS)} features | volume {use_vol}')
    for y, s, e in FOLDS:
        t = D[(D.Date >= s) & (D.Date <= e)]
        if len(t): print(f'   {y}: {len(t)} sessions, up share {t.y[~t.zero].mean():.4f}')
    run_classical(D, FEATS, os.path.join(pred_dir, 'predictions_classical.csv'))
    for arch in ARCHS:
        run_dl(D, FEATS, arch, os.path.join(pred_dir, f'predictions_dl_{arch}.csv'), mk)
    SUMMARY[mk] = inference(mk, out_dir, len(FEATS), use_vol)
    print(f'{mk} done, elapsed {(time.time()-t_all)/60:.1f} min')

# %%
# ---------------- cross-market tables ----------------
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
TA = pd.concat([v[0] for v in SUMMARY.values()]); TB = pd.concat([v[1] for v in SUMMARY.values()])
TA.to_csv(os.path.join(OUT_ROOT, 'all_markets_tableA_skill.csv'), index=False)
TB.to_csv(os.path.join(OUT_ROOT, 'all_markets_tableB_contrasts.csv'), index=False)
wide = TA.pivot_table(index='method', columns='market', values='bacc_mean', sort=False)[MARKETS]
wide.round(2).to_csv(os.path.join(OUT_ROOT, 'all_markets_bacc_mean.csv'))
print('Four-year mean balanced accuracy (%)'); print(wide.round(2).to_string())
sig = TB[(TB.p_daily_holm < 0.05) | (TB.p_year_holm < 0.05)]
print('\nContrasts significant after Holm adjustment:'); print(sig[['market', 'family', 'A', 'B', 'delta_daily', 'p_daily_holm', 'p_year_holm', 'mde80']].round(3).to_string() if len(sig) else 'none')
print('\nUp share and MDE by market:')
print(pd.DataFrame({mk: dict(up_share=v[2]['up_share'], n=v[2]['n'], features=v[2]['n_features'],
                             mde80_range=f"{v[1].mde80.min():.1f}-{v[1].mde80.max():.1f}") for mk, v in SUMMARY.items()}).T)

# %%
# ---------------- figures: one panel per market + combined ----------------
import matplotlib; import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 9})
lab = {'Naive': 'Naive rules', 'Classical': 'Classical ML', 'DL-R': 'DL, level regression', 'DL-C': 'DL, direct BCE'}
col = {'Naive': '#2a78d6', 'Classical': '#eb6834', 'DL-R': '#1baf7a', 'DL-C': '#eda100'}
mk_ = {'Naive': 'o', 'Classical': 's', 'DL-R': 'D', 'DL-C': '^'}
NAMES = {'SET50': 'SET50', 'SET100': 'SET100', 'SPX': 'S&P 500', 'CSI300': 'CSI 300'}
os.makedirs(os.path.join(OUT_ROOT, 'figures'), exist_ok=True)
fig, axes = plt.subplots(1, len(MARKETS), figsize=(3.0*len(MARKETS) + 1.6, 5.0), sharey=True, gridspec_kw={'wspace': 0.08})
axes = np.atleast_1d(axes)
for ax, mk in zip(axes, MARKETS):
    P = SUMMARY[mk][0].iloc[::-1].reset_index(drop=True)
    for f in lab:
        s = P[P.family == f]
        if s.empty: continue
        ax.errorbar(s.bacc_pooled, s.index, xerr=[s.bacc_pooled - s.ci_lo, s.ci_hi - s.bacc_pooled], fmt=mk_[f], color=col[f],
                    ms=4, lw=1.0, capsize=1.5, label=lab[f])
        ax.scatter(s.bacc_mean, s.index, marker='|', color='#444', s=30, zorder=3)
    ax.axvline(50, color='#888', lw=0.8, ls='--')
    ax.set_title(f"{NAMES[mk]} (Up share {100*SUMMARY[mk][2]['up_share']:.1f}%)", fontsize=9, loc='left')
    ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='x', color='#e5e5e5', lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel('Balanced accuracy (%)')
axes[0].set_yticks(np.arange(len(P)))
axes[0].set_yticklabels(P.method.str.replace(' (level-regression)', ' (LR)').str.replace(' (direct-BCE)', ' (BCE)'))
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc='lower center', ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.05))
fig.suptitle('Pooled seed-ensemble balanced accuracy, 2022-2025, with 95% circular block-bootstrap intervals', fontsize=9, x=0.02, ha='left')
for ext, dpi in [('png', 600), ('pdf', None)]:
    plt.savefig(os.path.join(OUT_ROOT, 'figures', f'fig_benchmark_all_markets.{ext}'), dpi=dpi, bbox_inches='tight')
plt.show()

# %%
# ---------------- download results (no raw prices) ----------------
zip_path = shutil.make_archive(os.path.join(ROOT, f'results_{TAG}'), 'zip', OUT_ROOT)   # saved in Google Drive
print('created', zip_path)
try:
    from google.colab import files; files.download(zip_path)
except Exception:
    pass
