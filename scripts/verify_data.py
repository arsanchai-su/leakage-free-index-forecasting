"""Check local data files against data/manifest.json (SHA-256, rows, date range)."""
import hashlib, json, os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
man = json.load(open(os.path.join(DATA, 'manifest.json')))
ok = True
for f in man['files']:
    p = os.path.join(DATA, f['name'])
    if not os.path.exists(p):
        print(f'MISSING  {f["name"]}'); ok = False; continue
    h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    d = pd.read_csv(p, parse_dates=['Date'])
    same = h == f['sha256']
    print(f"{'OK      ' if same else 'CHANGED '}{f['name']:20s} rows={len(d):5d} {d.Date.min().date()} -> {d.Date.max().date()}")
    ok &= same
print('All files match the manifest.' if ok else 'Some files are missing or differ; results may not reproduce exactly.')
