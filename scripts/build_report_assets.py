"""Regenerate publication figures and TeX tables from included aggregate results."""
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
ROOT=Path(__file__).resolve().parents[1]
LABELS={'joint_action_size_surprise':'Joint action/size','latent_player_archetype_shrinkage':'Player archetype','multiway_target_response':'Multiway response','opponent_bluff_likelihood':'Bluff likelihood','player_strategy_drift':'Strategy drift','postflop_behavior':'Postflop behavior','showdown_aware_range_calibration':'Showdown calibration','supported_fold_response_bluff_opportunity':'Fold response','within_hand_line_surprise':'Within-hand line'}
FAMILIES={'lightgbm':'LightGBM','catboost':'CatBoost','xgboost':'XGBoost','extra_trees':'Extra Trees'}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'report');args=ap.parse_args();out=args.output
    for name in ['figures','tables']:(out/name).mkdir(parents=True,exist_ok=True)
    d=json.loads((ROOT/'results/layer_metrics.json').read_text());rr=d['layers'];y=np.arange(len(rr))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(1,2,figsize=(10,4.25),sharey=True,gridspec_kw={'wspace':.15})
    for a,field,ci,title in [(ax[0],'improvement_prior','interval_prior','Against frozen prior'),(ax[1],'improvement_postflop','interval_postflop','Against selected postflop predictor')]:
        x=np.array([r[field] for r in rr]);iv=np.array([r[ci] for r in rr]);a.errorbar(x,y,xerr=[x-iv[:,0],iv[:,1]-x],fmt='o',color='#225e91',ecolor='#225e91',capsize=3,ms=4);a.axvline(0,color='#555555',lw=.8);a.set_title(title,fontsize=10);a.set_xlabel('Paired log-loss improvement');a.grid(axis='x',alpha=.18);a.set_axisbelow(True)
    for a in ax:a.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax[0].set_yticks(y,[LABELS[r['layer']] for r in rr]);ax[0].invert_yaxis();fig.subplots_adjust(left=.21,right=.985,bottom=.15,top=.9);fig.savefig(out/'figures/layer_comparisons.pdf');plt.close(fig)
    rows=['\\begin{table}[htbp]','\\centering\\small','\\caption{Selected models on confirmation support. Accuracy and loss are predictive measures, not win rates. All comparisons use 23 game clusters.}','\\label{tab:layers}','\\begin{tabular}{llrrrr}','\\toprule','Layer & Model & Rows & Accuracy & Log loss & $\\Delta$ prior \\\\','\\midrule']
    for r in rr:rows.append(f"{LABELS[r['layer']]} & {FAMILIES[r['family']]} & {r['confirmation_rows']:,} & {100*r['accuracy']:.2f}\\% & {r['log_loss']:.4f} & {r['improvement_prior']:.4f} \\\\")
    rows+=['\\bottomrule','\\end{tabular}','\\end{table}'];(out/'tables/layers.tex').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    h=json.loads((ROOT/'results/history_ablations.json').read_text());rs=[r for r in h['comparisons'] if r['comparison']=='session_vs_player'];roles=sorted(set(r['role'] for r in rs));fig,ax=plt.subplots(1,2,figsize=(8,2.8),sharey=True)
    for a,role in zip(ax,roles):
        data=sorted([r for r in rs if r['role']==role],key=lambda x:int(x['window']));x=np.arange(3);means=np.array([r['improvement'] for r in data]);lower=np.array([r['lower_bound'] for r in data]);a.vlines(x,lower,means,color='#225e91',lw=1.5);a.scatter(x,means,color='#225e91',s=28,label='Estimate');a.scatter(x,lower,marker='_',color='#225e91',s=90,label='Stored lower bound');a.axhline(0,color='#555555',lw=.8);a.set_xticks(x,[r['window']+'%' for r in data]);a.set_xlabel('Training window');a.set_title('Conditional sizing' if 'size' in role else 'Observed action');a.set_xlim(-.5,2.5);a.grid(axis='y',alpha=.18)
    ax[0].set_ylabel('Log-loss improvement');ax[1].legend(frameon=False,fontsize=8,loc='lower right');fig.tight_layout();fig.savefig(out/'figures/history_ablations.pdf');plt.close(fig)
    methods=json.loads((ROOT/'results/methods_evidence.json').read_text())
    training=[r'\begin{table}[!htbp]',r'\centering\small',
              r'\caption{Installed refits. Steps are retained boosting iterations or trees for Extra Trees. Time is the receipt elapsed time for each job, not additive wall-time or total project compute.}',
              r'\label{tab:training}',r'\begin{tabular}{llrrrr}',r'\toprule',
              r'Layer & Family & Features & Steps & $T$ & Seconds \\',r'\midrule']
    for r in methods['models']:
        training.append(f"{LABELS[r['layer']]} & {FAMILIES[r['family']]} & {r['feature_count']} & {r['best_iteration']:,} & {r['refit_temperature']:.1f} & {r['elapsed_seconds']:,.0f} \\\\")
    training += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (out/'tables/training.tex').write_text('\n'.join(training)+'\n',encoding='utf-8')
    rates=json.loads((ROOT/'results/stackhud_win_rate.json').read_text())
    games=[g for g in rates['games'] if g['included']]
    cumulative=np.cumsum([0]+[g['net_bb'] for g in games])
    fig,ax=plt.subplots(figsize=(8,2.8))
    ax.plot(np.arange(len(cumulative)),cumulative,color='#225e91',lw=1.6)
    ax.scatter(len(games),cumulative[-1],color='#225e91',s=25)
    ax.axhline(0,color='#777777',lw=.8);ax.grid(axis='y',alpha=.18)
    ax.set_xlabel('Eligible game (chronological order)');ax.set_ylabel('Cumulative net BB')
    ax.set_title(f"{rates['net_bb']:+.2f} BB / {rates['playing_hours']:.2f} corrected hours = {rates['bb_per_playing_hour']:+.2f} BB/hour",fontsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True));fig.tight_layout()
    fig.savefig(out/'figures/win_rate.pdf');plt.close(fig)
    print('Generated three PDF figures and two tables in',out)
if __name__=='__main__':main()
