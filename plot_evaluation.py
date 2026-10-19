from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parent;df=pd.read_csv(r/'results/model-comparison.csv')
plt.rcParams.update({'figure.facecolor':'#f4f5ee','axes.facecolor':'#f4f5ee','font.family':'DejaVu Sans','font.size':11,'axes.spines.right':False,'axes.spines.top':False,'text.color':'#214b3b','axes.labelcolor':'#214b3b'})
fig,ax=plt.subplots(figsize=(9,5.7));y=np.arange(len(df));ax.barh(y,df.cv_balanced_accuracy,xerr=df.cv_fold_sd,height=.45,color='#8cae96',capsize=4,label='Training CV mean ± fold SD');ax.scatter(df.holdout_balanced_accuracy,y,color='#a46c35',s=70,zorder=3,label='Held-out split');ax.set(yticks=y,yticklabels=df.model,xlim=(0,1.02),xlabel='Balanced accuracy',title='A reproducible comparison, with uncertainty');ax.legend(loc='lower left',frameon=False);fig.text(.08,.02,'New portfolio evaluation • fixed stratified split • 5-fold training CV • not clinical validation',fontsize=8,color='#5b6f59');fig.tight_layout(rect=(0,.04,1,1));fig.savefig(r/'results/model-comparison.png',dpi=170,bbox_inches='tight');plt.close(fig)
print('Evaluation chart created from measured results.')
