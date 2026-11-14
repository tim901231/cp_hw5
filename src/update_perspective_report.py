"""Export latest perspective normals/depth for report Figures 19--22."""
import shutil
import json
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import perspective as p
from entropy import load_images
from perspective_integration import from_normals
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/perspective_depth_diagnostics'
OUT.mkdir(parents=True, exist_ok=True)
from cp_hw5 import load_sources


def update_figures(root):
    out=root/'output/perspective_depth_diagnostics/report_results';out.mkdir(exist_ok=True)
    figures=root/'report/figures';records={}
    for name in ['face','cup','shoes']:
        if name=='face':
            d=np.load(root/'output/perspective_depth_diagnostics/improved/reconstruction.npz');g=d['log_distance'];normals=d['normals'];mask=d['foreground'];f=float(d['focal'])
            old=np.load(root/'output/bonus3_2_focal/selected/reconstruction.npz');B=old['B'];h,w=mask.shape
            smooth=cv2.GaussianBlur(B.T.reshape(h,w,3),(0,0),8)
            bu=np.gradient(smooth,axis=1).reshape(-1,3).T;bv=np.gradient(smooth,axis=0).reshape(-1,3).T
            inner=cv2.erode(mask.astype(np.uint8),np.ones((13,13),np.uint8))>0
            P=p.build_P(B,bu,bv,d['u'],d['v'],149.8431117)
            recovered,_,_=p.solve_ambiguity(P[(inner&old['inliers']).ravel()],B);recovered[2]*=149.8431117/f
            albedo=np.linalg.norm(recovered,axis=0).reshape(mask.shape)
            normal_source='sigma=8, 6px erosion, training-reference focal/normal rescaling'
        else:
            d=np.load(root/'output/capture_tests'/name/'perspective/reconstruction.npz');normals=d['normals'];mask=d['foreground'];f=4680.851063829787
            g,_=from_normals(normals,d['u'],d['v'],f,mask,3);albedo=d['albedo']
            normal_source='Existing perspective normals retained; integration updated only'
        distance=np.exp(g-np.median(g[mask]));lo,hi=np.percentile(distance[mask],[1,99])
        fig,axes=plt.subplots(1,3,figsize=(12,4.5))
        scale=max(np.percentile(albedo[mask],99.5),1e-12)
        axes[0].imshow(np.where(mask,albedo/scale,0),cmap='gray',vmin=0,vmax=1);axes[0].set_title('Albedo')
        axes[1].imshow(np.where(mask[...,None],(normals+1)/2,0));axes[1].set_title('Normals')
        axes[2].imshow(np.where(mask,distance,np.nan),cmap='gray',vmin=lo,vmax=hi);axes[2].set_title('Depth: black near, white far')
        for ax in axes:ax.axis('off')
        fig.tight_layout();filename='perspective_face.png' if name=='face' else name+'_perspective.png';fig.savefig(figures/filename,dpi=160);plt.close(fig)
        np.savez_compressed(out/(name+'.npz'),normals=normals,albedo=albedo,log_distance=g,distance=distance,foreground=mask,u=d['u'],v=d['v'],focal=f)
        records[name]={'normal_source':normal_source,'integration':'3 Huber iterations on incidence-weighted foreground edges','log_depth_span':float(np.ptp(g[mask]))}
    # Keep the preceding normal-comparison figure consistent with the new face result.
    face=np.load(out/'face.npz');mask=face['foreground'];h,w=mask.shape
    I=load_images(root/'data').reshape(7,-1);ref=np.linalg.lstsq(load_sources(),I,rcond=None)[0];ref/=np.linalg.norm(ref,axis=0);ref=ref.T.reshape(h,w,3)
    if np.mean(np.sum(ref*face['normals'],axis=-1)[mask])<0:ref=-ref
    initial=np.load(root/'output/bonus3_2/reconstruction.npz')['normals']
    fig,axes=plt.subplots(1,3,figsize=(12,5))
    for ax,n,title in zip(axes,[initial,face['normals'],ref],['Original: f=1, sigma=8','Reference-assisted: sigma=8','Known-light reference']):
        ax.imshow(np.where(mask[...,None],(n+1)/2,0));ax.set_title(title);ax.axis('off')
    fig.tight_layout();fig.savefig(figures/'perspective_normals.png',dpi=160);plt.close(fig)
    (out/'results.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2))


if __name__=='__main__':update_figures(ROOT)
