"""Reference-assisted perspective reconstruction comparison.
Select among a small, fixed grid using training normal references and positive
view-ray consistency. Held-out normal references are used only for reporting.
Focal/center are effective parameters, not independently calibrated intrinsics.
"""
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
from scipy.optimize import minimize_scalar
import shutil

if __name__=='__main__':
    base=np.load(ROOT/'output/bonus3_2_focal/selected/reconstruction.npz');B=base['B'];mask=base['foreground'];inliers=base['inliers'];h,w=mask.shape
    I=load_images(ROOT/'data').reshape(7,-1);ref=np.linalg.lstsq(load_sources(),I,rcond=None)[0];ref/=np.linalg.norm(ref,axis=0)
    yy,xx=np.indices(mask.shape);train=inliers & ((xx//16+yy//16)%2==0);test=inliers & ~train
    records=[];candidates=[];u,v,_,_=p.coordinates((h,w),184.,310.);fbase=149.8431117
    for sigma in [6,8,10]:
        smooth=cv2.GaussianBlur(B.T.reshape(h,w,3),(0,0),sigma)
        bu=np.gradient(smooth,axis=1).reshape(-1,3).T;bv=np.gradient(smooth,axis=0).reshape(-1,3).T
        P=p.build_P(B,bu,bv,u,v,fbase)
        for radius in [6,3*sigma]:
            inner=cv2.erode(mask.astype(np.uint8),np.ones((2*radius+1,)*2,np.uint8))>0
            rec,psi,s=p.solve_ambiguity(P[(inner&inliers).ravel()],B)
            def scaled(logf):
                n=rec.copy();n[2]*=fbase/np.exp(logf);return n/np.linalg.norm(n,axis=0)
            def error(logf,region):
                dot=np.sum(scaled(logf)*ref,axis=0);dot*=np.sign(np.mean(dot[train.ravel()]))
                return float(np.mean(np.degrees(np.arccos(np.clip(dot[region.ravel()],-1,1)))))
            opt=minimize_scalar(lambda f:error(f,train),bounds=(np.log(1),np.log(30000)),method='bounded')
            focal=float(np.exp(opt.x));n=scaled(opt.x).T.reshape(h,w,3)
            dotray=u*n[...,0]+v*n[...,1]+focal*n[...,2]
            if np.median(dotray[mask])<0:n=-n;dotray=-dotray
            negative=float(np.mean(dotray[mask]<0))
            g,_=from_normals(n,u,v,focal,mask,3)
            report={'sigma':sigma,'erosion_radius':radius,'focal':focal,'cx':184.,'cy':310.,'training_normal_mae':opt.fun,'validation_normal_mae':error(opt.x,test),'negative_ray_fraction':negative,'nullspace_gap':float(s[-2]/s[-1]),'log_depth_span':float(np.ptp(g[mask]))}
            records.append(report);candidates.append((report,n,g));print(report,flush=True)
    eligible=[c for c in candidates if c[0]['negative_ray_fraction']<.01]
    if not eligible:raise RuntimeError('No candidate satisfies positive-view-ray criterion')
    selected=min(eligible,key=lambda c:c[0]['training_normal_mae'])
    report,n,g=selected;f=report['focal'];distance=np.exp(g);oldg=base['log_distance'];oldd=np.exp(oldg-np.median(oldg[mask]))
    selected_out=OUT/'improved';selected_out.mkdir(exist_ok=True)
    np.savez_compressed(selected_out/'reconstruction.npz',normals=n,log_distance=g,distance=distance,foreground=mask,u=u,v=v,focal=f)
    # Show actual distance with black=near/white=far, using independent percentile display limits.
    fig,axes=plt.subplots(2,3,figsize=(12,9))
    panels=[((base['normals']+1)/2,'Previous normals'),((n+1)/2,'Improved normals'),(np.where(mask,base['log_distance'],np.nan),'Previous log-distance'),(np.where(mask,oldd,np.nan),'Previous distance'),(np.where(mask,distance,np.nan),'Improved distance'),(np.where(mask,g,np.nan),'Improved log-distance')]
    for ax,(im,title) in zip(axes.ravel(),panels):
        if im.ndim==3:ax.imshow(np.where(mask[...,None],im,0))
        else:
            lo,hi=np.percentile(im[mask],[1,99]);ax.imshow(im,cmap='gray',vmin=lo,vmax=hi)
        ax.set_title(title);ax.axis('off')
    fig.tight_layout();fig.savefig(selected_out/'comparison.png',dpi=160);plt.close(fig)
    shutil.copy2(selected_out/'comparison.png',ROOT/'report/figures/perspective_depth_improvement.png')
    fig=plt.figure(figsize=(11,5))
    for j,(dist,ff,title) in enumerate([(oldd,149.8431117,'Previous'),(distance,f,'Improved')]):
        ax=fig.add_subplot(1,2,j+1,projection='3d');ax.plot_surface(u*dist/ff,v*dist/ff,np.where(mask,-dist,np.nan),cmap='gray',rstride=3,cstride=3);ax.view_init(65,-90);ax.invert_yaxis();ax.set_axis_off();ax.set_title(title+' (arbitrary global scale)')
    fig.tight_layout();fig.savefig(selected_out/'surfaces.png',dpi=150);plt.close(fig)
    result={'selected':report,'all_candidates':records,'previous_validation_normal_mae':4.862419681879952,'previous_log_depth_span':float(np.ptp(oldg[mask])), 'selection':'Minimum training-reference angular error among candidates with fewer than 1% negative view-ray dots. Validation references are withheld from selection. No depth ground truth.', 'limitations':['Reference-assisted effective intrinsics, not fully uncalibrated or physically calibrated.','Three-step robust incidence-weighted integration is an extension, not the paper original solver.','Independent display ranges and arbitrary scale; cleaner appearance is not proof of accurate depth.']}
    (selected_out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SELECTED',report,flush=True)
