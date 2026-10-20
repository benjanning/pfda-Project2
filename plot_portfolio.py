"""Figures from saved predictions/metrics and authentic source features."""
from pathlib import Path
import json,numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parent;out=r/'results';df=pd.read_csv(out/'model-comparison.csv')
plt.rcParams.update({'figure.facecolor':'#f5f5ed','axes.facecolor':'#f5f5ed','font.family':'DejaVu Sans','font.size':11,'axes.spines.right':False,'axes.spines.top':False,'text.color':'#214b3b','axes.labelcolor':'#214b3b','xtick.color':'#536c5b','ytick.color':'#536c5b'})
def save(fig,name):
    fig.savefig(out/name,dpi=160,bbox_inches='tight',facecolor=fig.get_facecolor());plt.close(fig)
fig,ax=plt.subplots(figsize=(10.5,6));y=np.arange(len(df))
ax.barh(y,df.cv_balanced_accuracy,xerr=df.cv_fold_sd,height=.43,color='#83a68b',capsize=4,label='Training CV mean ± fold SD')
ax.scatter(df.holdout_balanced_accuracy,y,color='#ad7841',s=90,zorder=3,label='Reused exploratory holdout')
positive_total=int(pd.read_csv(out/'holdout-predictions.csv').actual.sum())
for i,row in df.iterrows():ax.text(.03,i+.29,f'{int(row.false_negatives)} false negatives / {positive_total} malignant holdout examples',fontsize=9,color='#6b755f')
ax.set(yticks=y,yticklabels=df.model,xlim=(0,1.03),xlabel='Balanced accuracy',title='Compare the models. Keep the errors in view.')
fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.55,.065),ncol=2,frameon=False,fontsize=9);fig.text(.18,.02,'Fixed stratified split · 5-fold training CV · fold SD is not a confidence interval · no clinical validation',fontsize=8,color='#6a7566');fig.tight_layout(rect=(0,.16,1,1));save(fig,'model-comparison.png')
matrices=json.loads((out/'confusion-matrices.json').read_text());fig,axes=plt.subplots(1,3,figsize=(13,4.7))
for ax,(name,item) in zip(axes,matrices.items()):
    matrix=np.array(item['counts']);ax.imshow(matrix,cmap='Greens',vmin=0,vmax=max(107,matrix.max()))
    for i in range(2):
        for j in range(2):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=25,color='white' if matrix[i,j]>55 else '#234b36',weight='bold')
    ax.set(xticks=[0,1],xticklabels=['Benign','Malignant'],yticks=[0,1],yticklabels=['Benign','Malignant'],xlabel='Predicted label',ylabel='Actual label',title=name)
    ax.text(.5,-.28,f'False negatives: {matrix[1,0]} of {matrix[1].sum()}',transform=ax.transAxes,ha='center',fontsize=10,color='#976e3e')
fig.suptitle('The same 171 held-out examples. Three different error patterns.',fontsize=15,y=1.03);fig.text(.5,-.05,'Educational comparison on a reused holdout. Lower-left cells are malignant examples predicted benign. Not a diagnostic system.',ha='center',fontsize=9,color='#687464');fig.tight_layout();save(fig,'confusion-matrices.png')
data=pd.read_csv(r/'Data/data.csv');fig,axes=plt.subplots(1,2,figsize=(11,5.5))
for ax,feature,label in zip(axes,['radius_mean','texture_mean'],['Mean radius','Mean texture']):
    bins=np.histogram_bin_edges(data[feature],bins=22)
    for value,title,color in [('B','Benign label','#376e50'),('M','Malignant label','#bf965c')]:
        values=data.loc[data.diagnosis==value,feature];ax.hist(values,bins=bins,alpha=.7,color=color,label=f'{title} (n={len(values)})')
    ax.set(xlabel=label+' (dataset feature)',ylabel='Number of examples');ax.legend(frameon=False,fontsize=9)
fig.suptitle('Explore the observations before interpreting the predictions.',fontsize=15);fig.text(.5,.01,'Descriptive plots of the complete public dataset; not feature selection or independent performance evidence.',ha='center',fontsize=9,color='#687464');fig.tight_layout(rect=(0,.05,1,.95));save(fig,'feature-distributions.png')
print('Three figures generated from saved measured results and source data.')
