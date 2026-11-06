import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, json, os
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':8})
_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA=os.environ.get('DATA',os.path.join(_ROOT,'data')); RES=os.environ.get('RES',os.path.join(_ROOT,'results','pricevol'))
M=['SET50','SET100','SPX','CSI300']; MN={'SET50':'SET50','SET100':'SET100','SPX':'S&P 500','CSI300':'CSI 300'}
# --- Figure: index levels with test years ---
fig,axes=plt.subplots(2,2,figsize=(7.2,4.6)); 
for ax,m in zip(axes.ravel(),M):
    d=pd.read_csv(f'{DATA}/{m}_days.csv',parse_dates=['Date']); d=d[d.Date<='2025-12-31']
    ax.plot(d.Date,d.Close,color='#1d2329',lw=0.7)
    j=json.load(open(f'{RES}/{m}/misc.json'))
    top=d.Close.max()*1.18; ax.set_ylim(d.Close.min()*0.85, top)
    for y in (2022,2023,2024,2025):
        ax.axvspan(pd.Timestamp(f'{y}-01-01'),pd.Timestamp(f'{y}-12-31'),color='#eb6834',alpha=0.10 if y%2 else 0.18,lw=0)
        ax.text(pd.Timestamp(f'{y}-07-01'),top*0.97,f"{100*j['up_share_by_year'][str(y)]:.0f}",ha='center',va='top',fontsize=6.5,color='#b04a17')
        tmax=d[d.Date<f'{y}-01-01'].Close.max()
        ax.plot([pd.Timestamp(f'{y}-01-01'),pd.Timestamp(f'{y}-12-31')],[tmax,tmax],color='#2a78d6',lw=0.9,ls='--')
    ax.set_title(f"{MN[m]}",fontsize=8.5,loc='left'); ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#e5e5e5',lw=0.5); ax.set_axisbelow(True); ax.tick_params(labelsize=7)
axes[1,0].set_ylabel('Index close'); axes[0,0].set_ylabel('Index close')
from matplotlib.lines import Line2D; from matplotlib.patches import Patch
fig.legend([Line2D([],[],color='#1d2329',lw=0.8),Patch(color='#eb6834',alpha=0.18),Line2D([],[],color='#2a78d6',ls='--')],
           ['daily close','test year (number: % of up sessions)','training-window maximum'],loc='lower center',ncol=3,frameon=False,fontsize=7.5,bbox_to_anchor=(0.5,-0.03))
fig.tight_layout(rect=(0,0.04,1,1))
for ext,dpi in [('png',600),('pdf',None)]: fig.savefig(f'{RES}/figures/fig_index_levels.{ext}',dpi=dpi,bbox_inches='tight')
plt.close(fig)
# --- Figure: per-year balanced accuracy ---
A=pd.read_csv(f'{RES}/all_markets_tableA_skill.csv'); T=pd.read_csv(f'{RES}/all_markets_tuned_skill.csv')
S=[('Persistence',A,'#5f6670','o','-'),('Logistic-L2',A,'#2a78d6','s','-'),('LSTM (level-regression)',A,'#1baf7a','D','-'),
   ('LSTM-Attention (level-regression)',A,'#127a55','v','-'),('Random forest (tuned)',T,'#eb6834','^','--'),('LSTM-Attention (tuned)',T,'#eda100','P','--')]
lab={'LSTM (level-regression)':'LSTM (LR)','LSTM-Attention (level-regression)':'LSTM-Attention (LR)'}
yrs=[2022,2023,2024,2025]
fig,axes=plt.subplots(1,4,figsize=(7.2,2.7),sharey=True)
for ax,m in zip(axes,M):
    for name,D,c,mk,ls in S:
        r=D[(D.market==m)&(D.method==name)].iloc[0]
        ax.plot(yrs,[r[f'bacc_{y}'] for y in yrs],color=c,marker=mk,ms=3.5,lw=1.0,ls=ls,label=lab.get(name,name))
    ax.axhline(50,color='#888',lw=0.8,ls=':'); ax.set_title(MN[m],fontsize=8.5,loc='left')
    ax.set_xticks(yrs); ax.set_xticklabels(["'22","'23","'24","'25"]); ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#e5e5e5',lw=0.5); ax.set_axisbelow(True)
axes[0].set_ylabel('Balanced accuracy (%)')
h,l=axes[0].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',ncol=3,frameon=False,fontsize=7.5,bbox_to_anchor=(0.5,-0.13))
fig.tight_layout()
for ext,dpi in [('png',600),('pdf',None)]: fig.savefig(f'{RES}/figures/fig_yearly_bacc.{ext}',dpi=dpi,bbox_inches='tight')
print('ok')
