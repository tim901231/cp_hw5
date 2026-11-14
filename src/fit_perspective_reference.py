"""Reproduce the reference-assisted perspective diagnostic (not uncalibrated).
Run from the assignment root after generating output/bonus3_2/reconstruction.npz.
Reference normals on alternating 16-pixel spatial blocks select parameters;
validation reference normals are withheld, but P uses all image inliers.
"""
import sys,numpy as np,cv2,json
from pathlib import Path
import perspective as m
from cp_hw5 import load_sources
def main():
    ROOT=Path(__file__).resolve().parents[1]
    out=ROOT/'output/bonus3_2_focal';out.mkdir(parents=True,exist_ok=True)
    d=np.load(ROOT/'output/bonus3_2/reconstruction.npz');B=d['B'];mask=d['inliers'];fg=d['foreground'];h,w=mask.shape
    u,v,cx,cy=m.coordinates((h,w))
    images=m.load_images(ROOT/'data');ref=np.linalg.lstsq(load_sources(),images.reshape(7,-1),rcond=None)[0];ref/=np.linalg.norm(ref,axis=0)
    # Deterministic spatial blocks: calibration references used only on training blocks.
    yy,xx=np.indices((h,w));train=mask & (((xx//16+yy//16)%2)==0);test=mask & ~train
    from scipy.optimize import minimize
    sigma=2
    smooth=cv2.GaussianBlur(B.T.reshape(h,w,3),(0,0),sigma)
    bu=np.gradient(smooth,axis=1).reshape(-1,3).T;bv=np.gradient(smooth,axis=0).reshape(-1,3).T
    base=m.build_P(B,bu,bv,u,v,1)[mask.ravel()];gram=base.T@base
    rng=np.random.default_rng(1);idx=rng.choice(np.flatnonzero(train),min(5000,train.sum()),replace=False)
    def cost(x):
     f,cxx,cyy=x
     T=np.eye(9);T[:3,6:]=(cxx-cx)/f*np.eye(3);T[3:6,6:]=(cyy-cy)/f*np.eye(3);T[6:,6:]/=f
     val,vec=np.linalg.eigh(T.T@gram@T);psi=vec[:,0].reshape(3,3).T
     try:n=np.linalg.solve(psi,B[:,idx]);n/=np.linalg.norm(n,axis=0)
     except np.linalg.LinAlgError:return 180
     dot=np.sum(n*ref[:,idx],axis=0);return np.mean(np.degrees(np.arccos(np.clip(dot*np.sign(dot.mean()),-1,1))))
    r=minimize(cost,[147.35393118656924,184,322.75],method='Nelder-Mead',bounds=[(20,5000),(0,w-1),(0,h-1)],options={'maxiter':300,'xatol':.01})
    f,cxx,cyy=r.x
    uu,vv,_,_=m.coordinates((h,w),cxx,cyy)
    P=m.build_P(B,bu,bv,uu,vv,f);rec,psi,s=m.solve_ambiguity(P[mask.ravel()],B);n=rec/np.linalg.norm(rec,axis=0)
    if np.mean(np.sum(n[:,train.ravel()]*ref[:,train.ravel()],axis=0))<0:n=-n
    err=np.degrees(np.arccos(np.clip(np.sum(n*ref,axis=0),-1,1)))
    report={'focal':float(f),'cx':float(cxx),'cy':float(cyy),'sigma':2,'train_mae_deg':float(err[train.ravel()].mean()),'validation_mae_deg':float(err[test.ravel()].mean()),'validation_median_deg':float(np.median(err[test.ravel()])),'train_pixels':int(train.sum()),'validation_pixels':int(test.sum()),'search_converged':bool(r.success),'nullspace_gap':float(s[-2]/s[-1]),'assumptions':'Effective parameters selected using known-light calibrated normals; not camera calibration or fully uncalibrated evaluation. Spatial 16-pixel validation blocks withheld from parameter selection; P uses all image inliers.'}
    print(json.dumps(report,indent=2));(out/'selected_parameters.json').write_text(json.dumps(report,indent=2))
    # Export pipeline to its own output folder, without overwriting f=1 outputs.
    import argparse
    m.run(argparse.Namespace(data=ROOT/'data',output=out/'selected',mask=None,focal=float(f),cx=float(cxx),cy=float(cyy),sigma=2.,ray_floor=1e-4))
    # Comparison uses a single global sign to match the selected output's ray convention.
    nout=np.load(out/'selected'/'reconstruction.npz')['normals']
    if np.mean(np.sum(nout.reshape(-1,3)[train.ravel()]*ref.T[train.ravel()],axis=1))<0:ref=-ref
    import matplotlib.pyplot as plt
    initial=d['normals'];fig,axs=plt.subplots(1,3,figsize=(12,5))
    for ax,im,title in zip(axs,[initial,nout,ref.T.reshape(h,w,3)],['Original: f=1, sigma=8','Fitted intrinsics, sigma=2','Known-light reference']):
     ax.imshow(np.where(fg[...,None],(im+1)/2,0));ax.set_title(title);ax.axis('off')
    fig.tight_layout();fig.savefig(out/'normal_comparison.png',dpi=160);plt.close(fig)


if __name__ == "__main__":
    main()
