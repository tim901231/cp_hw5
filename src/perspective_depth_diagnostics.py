"""Perspective depth diagnostics and confidence-weighted masked integration.
Run from the assignment root. Original reconstructions are never overwritten.
Confidence weights downweight near-grazing rays rather than clamping depth.
"""
import sys,json,importlib.util
from pathlib import Path
import numpy as np,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from perspective_integration import from_normals, weighted_integrate
from entropy import integrate_masked,load_images
import perspective as p
from cp_hw5 import integrate_poisson
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/perspective_depth_diagnostics';OUT.mkdir(exist_ok=True)


def synthetic():
    h,w=100,120;f=140.;u,v,_,_=p.coordinates((h,w));g=.16*np.sin(u/25)*np.cos(v/23)+.00002*u*u
    gu=.16/25*np.cos(u/25)*np.cos(v/23)+.00004*u;gv=-.16/23*np.sin(u/25)*np.sin(v/23)
    nn=np.stack([-f*gu,-f*gv,1+u*gu+v*gv]);nn/=np.linalg.norm(nn,axis=0)
    albedo=.6+.1*np.sin(u/17)*np.cos(v/19)
    rng=np.random.default_rng(3);lights=rng.normal(size=(12,3));lights[:,:2]*=.12;lights[:,2]=1;lights/=np.linalg.norm(lights,axis=1,keepdims=True)
    exact=lights@(nn*albedo).reshape(3,-1);mask=np.zeros((h,w),bool);mask[8:-8,8:-8]=True
    records=[]
    for noise in [0.,.001]:
        intensity=exact+rng.normal(0,noise,size=exact.shape)
        U,s,V=np.linalg.svd(intensity,full_matrices=False);B=np.sqrt(s[:3])[:,None]*V[:3]
        smooth=cv2.GaussianBlur(B.T.reshape(h,w,3),(0,0),1.)
        bu=np.gradient(smooth,axis=1).reshape(-1,3).T;bv=np.gradient(smooth,axis=0).reshape(-1,3).T
        rec,_,singular=p.solve_ambiguity(p.build_P(B,bu,bv,u,v,f)[mask.ravel()],B)
        normals=(rec/np.linalg.norm(rec,axis=0)).T.reshape(h,w,3)
        if np.mean(np.sum(normals*nn.transpose(1,2,0),axis=-1)[mask])<0:normals=-normals
        angle=np.degrees(np.arccos(np.clip(np.sum(normals*nn.transpose(1,2,0),axis=-1),-1,1)))
        recovered,_=from_normals(normals,u,v,f,mask)
        difference=recovered[mask]-g[mask];difference-=np.mean(difference)
        records.append({'noise_sigma':noise,'mean_normal_error_degrees':float(angle[mask].mean()),'log_depth_rmse':float(np.sqrt(np.mean(difference**2))),'true_log_depth_std':float(np.std(g[mask])),'nullspace_gap':float(singular[-2]/singular[-1])})
    print('Synthetic:',records,flush=True)
    return records


def real_data(name,path,f):
    d=np.load(path);n=d['normals'];mask=d['foreground'].copy();u=d['u'];v=d['v']
    gu,gv,valid,den=p.log_depth_gradients(n,u,v,f);valid &=mask
    methods={'Original':d['log_distance'],'Masked':integrate_masked(gu,gv,valid)}
    methods['Incidence weighted'],_=from_normals(n,u,v,f,mask)
    methods['Robust weighted'],history=from_normals(n,u,v,f,mask,6)
    fig,axes=plt.subplots(2,4,figsize=(15,8));metrics={}
    for col,(title,g) in enumerate(methods.items()):
        domain=mask&np.isfinite(g);lo,hi=np.percentile(g[domain],[1,99]);span=np.ptp(g[domain])
        distance=np.exp(g-np.median(g[domain]));dl,dh=np.percentile(distance[domain],[1,99])
        axes[0,col].imshow(np.where(domain,g,np.nan),cmap='gray',vmin=lo,vmax=hi)
        axes[1,col].imshow(np.where(domain,distance,np.nan),cmap='gray',vmin=dl,vmax=dh)
        axes[0,col].set_title(f'{title}\nlog-depth span {span:.3f}')
        axes[1,col].set_title('Distance (1-99% display)')
        for ax in axes[:,col]:ax.axis('off')
        metrics[title]={'log_depth_span':float(span),'distance_ratio':float(np.exp(min(span,700))),'coverage':float(domain.sum()/mask.sum())}
    fig.suptitle(name+': same normals and intrinsics; integration changes only');fig.tight_layout();fig.savefig(OUT/(name+'_comparison.png'),dpi=150);plt.close(fig)
    np.savez_compressed(OUT/(name+'_integration.npz'),mask=mask,normals=n,u=u,v=v,**{key.replace(' ','_'):value for key,value in methods.items()})
    print(name,metrics,flush=True)
    return metrics


if __name__=='__main__':
    report={'synthetic':synthetic()}
    report['face']=real_data('face',ROOT/'output/bonus3_2_focal/selected/reconstruction.npz',149.84311170054568)
    for name in ['cup','shoes']:
        report[name]=real_data(name,ROOT/'output/capture_tests'/name/'perspective/reconstruction.npz',4680.851063829787)
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
