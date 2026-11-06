"""AUC, log loss and block-bootstrap AUC intervals from the saved predictions (editor round).
Usage: python scripts/editor_analyses.py results/pricevol"""
import pandas as pd, numpy as np, glob, os, json
from sklearn.metrics import roc_auc_score, log_loss
from scipy import stats
import sys
RES=sys.argv[1] if len(sys.argv)>1 else 'results/pricevol'; OUT=os.path.join(RES,'editor'); os.makedirs(OUT,exist_ok=True)
MK=['SET50','SET100','SPX','CSI300']
rows=[]
for mk in MK:
    allp=pd.concat([pd.read_csv(f,parse_dates=['Date']) for f in glob.glob(f'{RES}/{mk}/predictions/predictions_*.csv')])
    allp=allp[~allp.zero]
    for m,g in allp.groupby('method'):
        fam=g.family.iloc[0]
        s=g.groupby('Date').agg(y=('y','first'),score=('score','mean'))
        if m=='Train-majority': auc=0.5
        else: auc=roc_auc_score(s.y,s.score)
        prob = fam in ('Classical','DL-C','Tuned','Extended') and m!='Ridge-return'
        ll = log_loss(s.y,np.clip(s.score,1e-6,1-1e-6)) if prob else np.nan
        base = log_loss(s.y, np.full(len(s), s.y.mean()))  # oracle base-rate log loss (test prevalence)
        rows.append(dict(market=mk,method=m,family=fam,auc=auc,logloss=ll,ll_base=base,n=len(s)))
A=pd.DataFrame(rows); A.to_csv(f'{OUT}/auc_logloss.csv',index=False)
pd.set_option('display.width',250)
print(A.pivot_table(index=['family','method'],columns='market',values='auc').round(3).to_string())
print(A[A.logloss.notna()].pivot_table(index=['family','method'],columns='market',values='logloss').round(4).to_string())
print('base ll', A.groupby('market').ll_base.first().round(4).to_dict())
# AUC CI for BCE collapse models via bootstrap? quick DeLong-free: bootstrap AUC for DL-C

# block-bootstrap 95% CI for AUC (block 10, 2000 reps), Holm over models within market for AUC>0.5 one-sided
from scipy.stats import rankdata
def fast_auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1*(n1+1)/2) / (n1*n0)
rng=np.random.default_rng(7); Bn=2000
out=[]
for mk in MK:
    allp=pd.concat([pd.read_csv(f,parse_dates=['Date']) for f in glob.glob(f'{RES}/{mk}/predictions/predictions_*.csv')]); allp=allp[~allp.zero]
    dates=None; S={}
    for m,g in allp.groupby('method'):
        if m=='Train-majority': continue
        s=g.groupby('Date').agg(y=('y','first'),score=('score','mean')).sort_index(); S[m]=s
    n=len(next(iter(S.values()))); k=int(np.ceil(n/10)); st=rng.integers(0,n,size=(Bn,k))
    IDX=((st[:,:,None]+np.arange(10)[None,None,:])%n).reshape(Bn,-1)[:,:n]
    for m,s in S.items():
        y=s.y.values; sc=s.score.values
        bs=np.array([fast_auc(y[i],sc[i]) for i in IDX])
        out.append(dict(market=mk,method=m,auc=fast_auc(y,sc),lo=np.percentile(bs,2.5),hi=np.percentile(bs,97.5),p_gt=float(np.mean(bs<=0.5)),p_lt=float(np.mean(bs>=0.5))))
C=pd.DataFrame(out)
def holm(p):
    p=np.asarray(p,float); o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for r,i in enumerate(o): run=max(run,(m-r)*p[i]); adj[i]=min(1,run)
    return adj
C['p_gt_holm']=C.groupby('market').p_gt.transform(lambda q: holm(q.values))
C['p_lt_holm']=C.groupby('market').p_lt.transform(lambda q: holm(q.values))
C.to_csv(f'{OUT}/auc_ci.csv',index=False)
print(C[(C.p_gt_holm<0.05)|(C.p_lt_holm<0.05)].round(3).to_string())
print(C[C.method.str.contains('direct-BCE')].round(3).to_string())

# upper 95% limits of network-minus-baseline differences (Table A10): delta + 1.96 * block-bootstrap SE
E = pd.read_csv(f'{RES}/revision2/equivalence_tost.csv'); X = pd.read_csv(f'{RES}/all_markets_extended_contrasts.csv')
nets = ['LSTM', 'CNN', 'LSTM-CNN', 'LSTM-Attention', 'LSTM-CNN-Attention']
tu = E[(E.arm == 'tuned') & (E.A.isin([f'{a} (tuned)' for a in nets]))].copy(); tu['arm'] = 'tuned DL'
fr = E[E.arm == 'frozen'].copy()
ex = X[X.B.isin(['Logistic-L2', 'Persistence'])].copy(); ex['arm'] = 'extended'
ex['family'] = ex.B.map({'Logistic-L2': 'E1 extended vs logistic', 'Persistence': 'E2 extended vs persistence'})
ex['delta'] = ex.delta_daily; ex['se'] = ex.se_daily
U = pd.concat([fr, tu, ex])[['arm', 'family', 'market', 'delta', 'se']]; U['hi'] = U.delta + 1.96 * U.se
N = U.groupby(['arm', 'family', 'market'], sort=False).agg(delta_max=('delta', 'max'), ci_hi_min=('hi', 'min'),
                                                           ci_hi_max=('hi', 'max'), ci_hi_median=('hi', 'median')).reset_index()
N.to_csv(f'{OUT}/ci_upper_bounds.csv', index=False)
c = U[U.market != 'SET50']; print('confirmatory upper limits: min %.1f, median %.1f, max %.1f' % (c.hi.min(), c.hi.median(), c.hi.max()))
