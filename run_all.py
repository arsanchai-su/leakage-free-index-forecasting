"""Run the full study locally (same code as notebooks/Run_4Markets_colab.ipynb).

Usage
    python run_all.py                      # all markets, frozen + tuned + leakage audit
    python run_all.py --quick              # smoke test: 1 seed, 2 epochs, 2 architectures
    python run_all.py --markets SET50 SPX  # subset of markets
    python run_all.py --skip-tuned --skip-leakage
    python run_all.py --skip-frozen --skip-tuned --skip-leakage --extended   # extended arm and revision analyses

Data: put the 12 Investing.com CSV files listed in data/manifest.json into data/ (see data/README.md).
Results are written to results/<tag>/ where tag is pricevol (default) or quick_pricevol.
"""
import argparse, os, re, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, 'src')


def code_cells(path):
    """Split a '# %%' script into code cells (markdown cells are skipped)."""
    text = open(path, encoding='utf-8').read()
    return [c.split('\n', 1)[1] for c in re.split(r'^# %%', text, flags=re.M)[1:] if not c.startswith(' [markdown]')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--markets', nargs='+', default=['SET50', 'SET100', 'SPX', 'CSI300'])
    ap.add_argument('--price-only', action='store_true', help='use the 75 price-only features in every market')
    ap.add_argument('--skip-tuned', action='store_true')
    ap.add_argument('--skip-leakage', action='store_true')
    ap.add_argument('--skip-frozen', action='store_true', help='load definitions only; reuse existing frozen-arm results')
    ap.add_argument('--extended', action='store_true', help='run the extended arm and the revision analyses')
    a = ap.parse_args()

    import matplotlib; matplotlib.use('Agg')
    g = {'__name__': '__main__'}
    cells = code_cells(os.path.join(SRC, 'benchmark.py'))
    cells[0] = (cells[0]
                .replace("MARKETS = ['SET50', 'SET100', 'SPX', 'CSI300']", f"MARKETS = {a.markets!r}")
                .replace("QUICK = False", f"QUICK = {a.quick}")
                .replace("USE_VOLUME = 'auto'", "USE_VOLUME = False" if a.price_only else "USE_VOLUME = 'auto'")
                .replace("LOCAL_DIR = '/content/Run_4Markets'", f"LOCAL_DIR = {ROOT!r}"))
    t0 = time.time()
    for c in cells:
        if 'make_archive' in c:
            continue  # Colab download cell
        if a.skip_frozen and any(c.startswith(h) for h in ('# ---------------- run all markets', '# ---------------- cross-market tables', '# ---------------- figures')):
            continue
        c = c.replace("f'/content/results_{TAG}'", "os.path.join(ROOT, f'results_{TAG}')").replace('plt.show()', 'plt.close("all")')
        exec(c, g)
    if not a.skip_tuned:
        for c in code_cells(os.path.join(SRC, 'tuned.py')):
            exec(c, g)
    if not a.skip_leakage:
        for c in code_cells(os.path.join(SRC, 'leakage.py')):
            exec(c.replace('plt.show()', 'plt.close("all")'), g)
    if a.extended:
        tuned_cells = code_cells(os.path.join(SRC, 'tuned.py'))
        if a.skip_tuned: exec(tuned_cells[0], g)          # definitions needed for the baselines
        for name in ('extended.py', 'revision2.py'):
            for c in code_cells(os.path.join(SRC, name)):
                exec(c.replace('plt.show()', 'plt.close("all")'), g)
    print(f'finished in {(time.time() - t0) / 60:.1f} min; results in {g["OUT_ROOT"]}')


if __name__ == '__main__':
    sys.exit(main())
