import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 8})
import os
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'pricevol', 'figures') + os.sep
BLUE, ORANGE, GREEN, AMBER, GREY, RED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#5f6670', '#c0392b'
LIGHT = {BLUE: '#e3eefb', ORANGE: '#fdebe2', GREEN: '#ddf4ea', AMBER: '#fdf2d6', GREY: '#eef0f2', RED: '#f8e1de'}


def box(ax, x, y, w, h, title, lines=(), c=GREY, fs=8, tfs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.012', fc=LIGHT[c], ec=c, lw=1.0))
    ax.text(x + w/2, y + h - 0.022, title, ha='center', va='top', fontsize=tfs, weight='bold', color='#1d2329')
    for i, ln in enumerate(lines):
        ax.text(x + w/2, y + h - 0.062 - i*0.034, ln, ha='center', va='top', fontsize=fs, color='#1d2329')


def arrow(ax, p, q, c=GREY, rad=0.0, lw=1.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=9, color=c, lw=lw, connectionstyle=f'arc3,rad={rad}'))


def save(fig, name):
    for ext, dpi in [('png', 600), ('pdf', None)]:
        fig.savefig(OUT + f'{name}.{ext}', dpi=dpi, bbox_inches='tight')
    plt.close(fig)

# ---------------- Diagram 1: study design ----------------
fig = plt.figure(figsize=(7.2, 4.3)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
box(ax, 0.02, 0.70, 0.20, 0.26, 'Data', ['SET50, SET100,', 'S&P 500, CSI 300', 'daily, weekly, monthly', '2012–2025', 'SHA-256 manifest'], GREY)
box(ax, 0.27, 0.70, 0.21, 0.26, 'Point-in-time contract', ['features known at close t', 'label: sign(Cₜ₊₁ − Cₜ)', 'last completed W/M bar', 'expanding window, purge', 'train-only scaling'], GREY)
box(ax, 0.53, 0.70, 0.19, 0.26, 'Features', ['85 technical variables', '75 without volume', '(S&P 500)', '60-session warm-up'], GREY)
box(ax, 0.77, 0.70, 0.21, 0.26, 'Outer folds', ['test 2022, 2023,', '2024, 2025', 'train 2012 → year − 1', '3,910 scored sessions'], GREY)
for a, b in [((0.22, 0.83), (0.27, 0.83)), ((0.48, 0.83), (0.53, 0.83)), ((0.72, 0.83), (0.77, 0.83))]: arrow(ax, a, b)
ax.plot([0.875, 0.875], [0.70, 0.655], color=GREY, lw=1.0); ax.plot([0.17, 0.875], [0.655, 0.655], color=GREY, lw=1.0)
box(ax, 0.02, 0.27, 0.30, 0.34, 'A. Frozen benchmark', ['Naive: majority, persistence,', 'reversal, MA20 trend', 'Classical: logistic, ridge,', 'random forest, LightGBM', 'Deep: LSTM, CNN, LSTM-CNN,', 'LSTM-Att., LSTM-CNN-Att.', '× {level regression, BCE}, 5 seeds'], BLUE, fs=7.6)
box(ax, 0.35, 0.27, 0.30, 0.34, 'B. Tuned arm', ['validation = last 250 training', 'sessions (purged)', 'class-balanced loss / weights', 'grid search (classical)', 'early stopping (deep)', 'refit on full training window', 'pre-specified ensemble'], GREEN, fs=7.6)
box(ax, 0.68, 0.27, 0.30, 0.34, 'C. Leakage audit', ['P0 leak-free reference', 'P1 causal rolling VMD', 'L1 full-sample VMD features', 'L2 VMD trend as label', 'L3 next-day open as feature', 'L4 100 − MAPE as "accuracy"', 'L5 = L1 + L3'], ORANGE, fs=7.6)
for x in (0.17, 0.50, 0.83): arrow(ax, (x, 0.655), (x, 0.61))
box(ax, 0.10, 0.02, 0.80, 0.19, 'Inference', ['balanced accuracy  •  Pesaran–Timmermann test  •  circular block bootstrap (B = 5,000)',
    'paired contrasts by year (exact sign-flip), quarter, and session  •  Holm adjustment  •  MDE₈₀ = 2.8 · SE'], AMBER, fs=7.8)
for x in (0.17, 0.50, 0.83): arrow(ax, (x, 0.27), (x if x != 0.83 else 0.80, 0.21))
save(fig, 'diag_design')

# ---------------- Diagram 2: folds and point-in-time ----------------
fig = plt.figure(figsize=(7.2, 3.9)); ax = fig.add_axes([0.0, 0.0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
x0, x1 = 0.12, 0.96; yr = lambda y: x0 + (y - 2012)/(2026 - 2012)*(x1 - x0)
ax.text(0.01, 0.94, 'A', fontsize=10, weight='bold')
for y in range(2012, 2027, 2):
    ax.plot([yr(y)]*2, [0.50, 0.92], color='#d5d9de', lw=0.6, zorder=0); ax.text(yr(y), 0.475, str(y), ha='center', fontsize=7, color=GREY)
for i, ty in enumerate([2022, 2023, 2024, 2025]):
    yy = 0.83 - i*0.085
    ax.text(0.105, yy + 0.02, f'Fold {i+1}', ha='right', va='center', fontsize=7.5)
    ax.add_patch(Rectangle((yr(2012), yy), yr(ty - 1) - yr(2012) - 0.004, 0.04, fc=LIGHT[BLUE], ec=BLUE, lw=0.8))
    ax.add_patch(Rectangle((yr(ty - 1) - 0.004, yy), 0.004, 0.04, fc=RED, ec=RED, lw=0))
    ax.add_patch(Rectangle((yr(ty - 1) - 0.004 - 0.017, yy), 0.017, 0.04, fc=LIGHT[GREEN], ec=GREEN, lw=0.8, hatch='////'))
    ax.add_patch(Rectangle((yr(ty), yy), yr(ty + 1) - yr(ty), 0.04, fc=LIGHT[ORANGE], ec=ORANGE, lw=0.8))
lg = [(LIGHT[BLUE], BLUE, '', 'training (expanding from 2012)'), (LIGHT[GREEN], GREEN, '////', 'validation, last 250 sessions (tuned arm only)'),
      (RED, RED, '', 'purged row'), (LIGHT[ORANGE], ORANGE, '', 'test year (scored once)')]
for i, (fc, ec, h, t) in enumerate(lg):
    lx = 0.12 + (i % 2)*0.42; ly = 0.42 - (i//2)*0.05
    ax.add_patch(Rectangle((lx, ly), 0.025, 0.028, fc=fc, ec=ec, hatch=h, lw=0.8)); ax.text(lx + 0.033, ly + 0.014, t, va='center', fontsize=7.5)
ax.text(0.01, 0.30, 'B', fontsize=10, weight='bold')
days = [0.20, 0.31, 0.42, 0.53, 0.64, 0.75]
lbl = ['t−4', 't−3', 't−2', 't−1', 't', 't+1']
for d, l in zip(days, lbl):
    ax.add_patch(Rectangle((d - 0.035, 0.17), 0.07, 0.07, fc=LIGHT[BLUE] if l != 't+1' else LIGHT[ORANGE], ec=BLUE if l != 't+1' else ORANGE, lw=0.8))
    ax.text(d, 0.205, l, ha='center', va='center', fontsize=8)
ax.text(0.42, 0.265, 'window of W sessions (W = 5, 10, or 20)', ha='center', fontsize=7.5, color=BLUE)
ax.plot([days[0] - 0.035, days[4] + 0.035], [0.255, 0.255], color=BLUE, lw=0.8)
ax.annotate('', xy=(0.75, 0.165), xytext=(0.64, 0.165), arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1, connectionstyle='arc3,rad=0.5'))
ax.text(0.80, 0.135, 'label yₜ = 1 if Cₜ₊₁ > Cₜ', ha='left', fontsize=7.5, color=ORANGE)
ax.text(0.82, 0.235, 'forecast made at\nclose of day t', fontsize=7.5, va='center')
ax.plot([0.695, 0.695], [0.15, 0.27], color=RED, lw=1, ls='--'); ax.text(0.695, 0.285, 'forecast origin', ha='center', fontsize=7, color=RED)
ax.add_patch(Rectangle((0.165, 0.02), 0.13, 0.05, fc=LIGHT[GREY], ec=GREY, lw=0.8)); ax.text(0.23, 0.045, 'week w−1', ha='center', va='center', fontsize=7)
ax.add_patch(Rectangle((0.305, 0.02), 0.13, 0.05, fc='white', ec=GREY, lw=0.8, ls='--')); ax.text(0.37, 0.045, 'week w (current)', ha='center', va='center', fontsize=6.6, color=GREY)
ax.add_patch(Rectangle((0.465, 0.02), 0.13, 0.05, fc=LIGHT[GREY], ec=GREY, lw=0.8)); ax.text(0.53, 0.045, 'month m−1', ha='center', va='center', fontsize=7)
ax.add_patch(Rectangle((0.605, 0.02), 0.13, 0.05, fc='white', ec=GREY, lw=0.8, ls='--')); ax.text(0.67, 0.045, 'month m (current)', ha='center', va='center', fontsize=6.6, color=GREY)
ax.text(0.76, 0.045, 'only completed (shaded)\nbars enter the features', va='center', fontsize=7)
save(fig, 'diag_folds')

# ---------------- Diagram 3: leakage mechanisms ----------------
fig = plt.figure(figsize=(7.2, 4.4)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
rows = [('P1  causal VMD', 'decomposes only sessions up to t', '50\u201353%', GREEN, (0.30, 0.62), None),
        ('L1  full-sample VMD', 'one decomposition of 2012–2025; modes at t contain t+1, t+2, …', '67–80%', ORANGE, (0.30, 0.85), None),
        ('L2  VMD trend as label', 'label = direction of a smoothed trend built from future prices', '53–67%', ORANGE, (0.30, 0.62), (0.62, 0.85)),
        ('L3  next-day open', 'opening price of t+1 is used as a feature at t', '62–68%', ORANGE, (0.30, 0.665), None),
        ('L4  100 − MAPE', 'level error of "tomorrow = today" reported as accuracy', '99%', RED, None, None)]
xT = 0.62
ax.text(0.30, 0.955, '2012', ha='center', fontsize=7.5, color=GREY); ax.text(xT, 0.955, 't (forecast origin)', ha='center', fontsize=7.5, color=RED)
ax.text(0.85, 0.955, '2025', ha='center', fontsize=7.5, color=GREY)
ax.plot([xT, xT], [0.05, 0.94], color=RED, lw=1, ls='--', zorder=0)
ax.text(0.02, 0.955, 'Variant', fontsize=8, weight='bold'); ax.text(0.975, 0.955, '', fontsize=8)
for i, (name, desc, acc, c, span, lab) in enumerate(rows):
    y = 0.80 - i*0.165
    ax.text(0.02, y + 0.035, name, fontsize=8.5, weight='bold', va='center')
    ax.text(0.02, y - 0.03, desc, fontsize=7.5, va='center', color='#333')
    if span:
        a, b = span
        ax.add_patch(Rectangle((a, y + 0.0), min(b, xT) - a, 0.05, fc=LIGHT[BLUE], ec=BLUE, lw=0.8))
        if b > xT: ax.add_patch(Rectangle((xT, y), b - xT, 0.05, fc=LIGHT[ORANGE], ec=ORANGE, lw=0.8, hatch='////'))
    if lab:
        ax.add_patch(Rectangle((lab[0], y + 0.06), lab[1] - lab[0], 0.025, fc=LIGHT[ORANGE], ec=ORANGE, lw=0.8, hatch='////'))
        ax.text(0.735, y + 0.0725, 'label window', ha='center', va='center', fontsize=6.5)
    if name.startswith('L4'):
        ax.text(0.44, y + 0.025, '|Ĉₜ₊₁ − Cₜ₊₁| / Cₜ₊₁ ≈ 0.8%', ha='center', va='center', fontsize=8)
        ax.text(0.75, y + 0.025, 'no model, no direction', ha='center', va='center', fontsize=8, color=RED)
    ax.add_patch(FancyBboxPatch((0.88, y + 0.003), 0.10, 0.045, boxstyle='round,pad=0.004', fc=LIGHT[c], ec=c, lw=0.9))
    ax.text(0.93, y + 0.0255, acc, ha='center', va='center', fontsize=8.5, weight='bold', color=c if c != GREEN else '#127a55')
ax.add_patch(Rectangle((0.30, 0.02), 0.03, 0.025, fc=LIGHT[BLUE], ec=BLUE, lw=0.8)); ax.text(0.335, 0.0325, 'information available at t', va='center', fontsize=7.5)
ax.add_patch(Rectangle((0.58, 0.02), 0.03, 0.025, fc=LIGHT[ORANGE], ec=ORANGE, hatch='////', lw=0.8)); ax.text(0.615, 0.0325, 'information from after t (leak)', va='center', fontsize=7.5)
ax.text(0.93, 0.92, 'Accuracy\n2022–2025', ha='center', fontsize=7.5, weight='bold')
save(fig, 'diag_leakage')
print('done')
