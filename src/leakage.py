# %% [markdown]
# # Leakage audit: how common protocol errors produce accuracies above 85%
#
# Each variant changes ONE thing in the leak-free protocol and is evaluated on the same 2022-2025 test sessions with the
# same learners (L2 logistic regression and LightGBM, fixed hyperparameters). The high scores are artefacts of the
# protocol error, not forecasting skill, and must be reported as such.
#
# | Code | Protocol error | What leaks |
# |---|---|---|
# | P0 | none (leak-free reference) | nothing |
# | P1 | causal rolling VMD features (250-day window, decomposed up to t only) | nothing (leak-free VMD) |
# | L1 | VMD fitted once on the full 2012-2025 series, modes used as features | future prices through the decomposition |
# | L2 | target = next-day direction of the full-sample VMD trend instead of the real close | smoothed future in the label |
# | L3 | next session's opening gap added as a feature | information published after the forecast time |
# | L4 | "accuracy" reported as 100 - MAPE of a next-close level forecast (persistence) | wrong metric, no skill needed |
# | L5 | L1 and L3 together | both of the above |

# %%
from vmdpy import VMD
K_MODES, ALPHA, ROLL = 6, 2000, 250


def vmd_modes(x):
    x = np.asarray(x, float); n = len(x) - len(x) % 2          # vmdpy needs even length
    u, _, _ = VMD(x[-n:], ALPHA, 0, K_MODES, 0, 1, 1e-7)
    out = np.full((len(x), K_MODES), np.nan); out[-n:] = u.T
    return out


def causal_vmd(close):
    c = np.asarray(close, float); out = np.full((len(c), K_MODES), np.nan)
    for t in range(ROLL - 1, len(c)):
        out[t] = vmd_modes(c[t - ROLL + 1:t + 1])[-1]
    return out


def fit_eval(D, cols, ycol='y'):
    res = {}
    for name in ['Logistic-L2', 'LightGBM']:
        ys, ps = [], []
        for year, s, e in FOLDS:
            tr = D[D.Date < s].iloc[:-1].dropna(subset=cols + [ycol]); te = D[(D.Date >= s) & (D.Date <= e)]
            te = te[~te.zero].dropna(subset=cols + [ycol])
            if name == 'Logistic-L2':
                sc = StandardScaler().fit(tr[cols]); m = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(tr[cols]), tr[ycol])
                p = m.predict(sc.transform(te[cols]))
            else:
                m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8,
                                       subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1).fit(tr[cols], tr[ycol])
                p = m.predict(te[cols])
            ys.append(te[ycol].values); ps.append(p)
        y, p = np.concatenate(ys), np.concatenate(ps)
        res[name] = dict(acc=100*(y == p).mean(), bacc=100*bacc(y, p), n=len(y))
    return res


LEAK = []
for mk in MARKETS:
    t0 = time.time()
    A = build_aligned(mk)
    use_vol = (A['Volume_D'].notna().all() and (A['Volume_D'] > 0).all()) if USE_VOLUME == 'auto' else False
    D, FEATS = build_features(A, use_vol)
    days = load(f'{mk}_days.csv').set_index('Date')
    D['next_gap'] = D.Date.map(days.Open.shift(-1)) / D.C - 1                     # L3: next session's open, known only tomorrow
    full = vmd_modes(D.C.values)                                                 # L1/L2: decomposition sees the whole sample
    vm_full = [f'vmdF_{k}' for k in range(K_MODES)]
    for k in range(K_MODES): D[vm_full[k]] = full[:, k] / D.C
    trend = full[:, 0] + full[:, 1]
    D['y_trend'] = (np.r_[trend[1:], np.nan] > trend).astype(float); D.loc[D.index[-1], 'y_trend'] = np.nan
    cache = os.path.join(OUT_ROOT, mk, 'causal_vmd.npy')
    if os.path.exists(cache): cv = np.load(cache)
    else: cv = causal_vmd(D.C.values); np.save(cache, cv)
    vm_c = [f'vmdC_{k}' for k in range(K_MODES)]
    for k in range(K_MODES): D[vm_c[k]] = cv[:, k] / D.C
    variants = {'P0 leak-free reference': (FEATS, 'y'), 'P1 causal rolling VMD': (FEATS + vm_c, 'y'),
                'L1 full-sample VMD features': (FEATS + vm_full, 'y'), 'L2 full-sample VMD trend as target': (FEATS, 'y_trend'),
                'L3 next-day opening gap as feature': (FEATS + ['next_gap'], 'y'),
                'L5 L1 + L3 combined': (FEATS + vm_full + ['next_gap'], 'y')}
    for v, (cols, ycol) in variants.items():
        for model, r in fit_eval(D, cols, ycol).items():
            LEAK.append(dict(market=mk, variant=v, model=model, **r))
    te = D[(D.Date >= FOLDS[0][1]) & (D.Date <= FOLDS[-1][2])]
    mape = 100*np.mean(np.abs(te.C_next - te.C)/te.C_next)                       # L4: persistence level forecast
    for model in ['Logistic-L2', 'LightGBM']:
        LEAK.append(dict(market=mk, variant='L4 "accuracy" = 100 - MAPE (persistence)', model=model, acc=100 - mape, bacc=np.nan, n=len(te)))
    print(f'{mk} leakage audit done in {(time.time()-t0)/60:.1f} min', flush=True)

LK = pd.DataFrame(LEAK); LK.to_csv(os.path.join(OUT_ROOT, 'all_markets_leakage_audit.csv'), index=False)
show = LK.pivot_table(index=['variant', 'model'], columns='market', values='acc', sort=False)[MARKETS]
print('Accuracy (%), 2022-2025 test sessions'); print(show.round(1).to_string())
print('\nBalanced accuracy (%)'); print(LK.pivot_table(index=['variant', 'model'], columns='market', values='bacc', sort=False)[MARKETS].round(1).to_string())

# %%
fig, ax = plt.subplots(figsize=(7.2, 3.6))
V = list(dict.fromkeys(LK.variant)); x = np.arange(len(V)); w = 0.8/len(MARKETS)
pal = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
for i, mk in enumerate(MARKETS):
    vals = [LK[(LK.market == mk) & (LK.variant == v) & (LK.model == 'LightGBM')].acc.iloc[0] for v in V]
    ax.bar(x + (i - (len(MARKETS)-1)/2)*w, vals, w, color=pal[i], label=NAMES[mk])
ax.axhline(50, color='#888', lw=0.8, ls='--'); ax.axhline(85, color='#b00', lw=0.8, ls=':')
ax.text(len(V) - 0.5, 86, '85%', color='#b00', fontsize=7, ha='right')
ax.set_xticks(x); ax.set_xticklabels([v.split(' ', 1)[0] for v in V]); ax.set_ylim(40, 101)
ax.set_ylabel('Accuracy on 2022-2025 test sessions (%)')
ax.set_title('P = leak-free protocol; L = one protocol error (LightGBM)', fontsize=9, loc='left')
ax.spines[['top', 'right']].set_visible(False); ax.legend(fontsize=7, frameon=False, ncol=4, loc='upper left')
for ext, dpi in [('png', 600), ('pdf', None)]:
    plt.savefig(os.path.join(OUT_ROOT, 'figures', f'fig_leakage_audit.{ext}'), dpi=dpi, bbox_inches='tight')
plt.show()
