"""Plot the exact entropy-search samples before and after GBR correction."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from entropy import histogram_entropy


def generate_distributions(root):
    datasets=[('Face',root/'output/bonus3/reconstruction.npz'),
              ('Cup',root/'output/capture_tests/cup/reconstruction.npz'),
              ('Boot',root/'output/capture_tests/shoes/reconstruction.npz')]
    fig,axes=plt.subplots(3,2,figsize=(11,10))
    metrics={}
    for row,(name,path) in enumerate(datasets):
        data=np.load(path)
        if 'sample_indices' in data:
            indices=data['sample_indices']
        else:
            indices=np.flatnonzero(data['inliers'])
            if len(indices)>12000:
                indices=np.random.default_rng(0).choice(indices,12000,replace=False)
        report={'samples':int(len(indices))}
        max_height=0
        for col,(key,label,color) in enumerate([('B_before','Before GBR','#627b98'),('B_after','After entropy GBR','#be6a3c')]):
            values=np.linalg.norm(data[key][:,indices],axis=0)
            counts,edges=np.histogram(values,bins=256,range=(values.min(),values.max()))
            probabilities=counts/len(values)
            score=histogram_entropy(values,256)
            ax=axes[row,col]
            ax.bar(edges[:-1],probabilities,width=np.diff(edges),align='edge',color=color,linewidth=0)
            ax.set_title(f'{name}: {label}\nEntropy = {score:.4f} nats; N = {len(indices):,}')
            ax.set_xlabel('Recovered albedo (arbitrary units)')
            ax.set_ylabel('Fraction of sampled pixels per bin')
            ax.grid(axis='y',alpha=.2)
            max_height=max(max_height,float(probabilities.max()))
            report['entropy_before' if col==0 else 'entropy_after']=float(score)
        for ax in axes[row]:ax.set_ylim(0,max_height*1.1)
        metrics[name.lower()]=report
    fig.tight_layout(h_pad=2)
    output=root/'report/figures/albedo_distributions.png'
    fig.savefig(output,dpi=180,bbox_inches='tight');plt.close(fig)
    (root/'report/albedo_distribution_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    generate_distributions(Path(__file__).resolve().parents[1])
