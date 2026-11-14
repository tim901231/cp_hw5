"""Generate report figures and numerical checks; run from the assignment root."""
import sys,json,shutil
from pathlib import Path
import numpy as np,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from entropy import load_images,recover_integrable,save_image,integrate_masked
from cp_hw5 import load_sources,integrate_poisson,integrate_frankot
def main():
    ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'report'/'figures';OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10})
    images=load_images(ROOT/'data');k,h,w=images.shape;I=images.reshape(k,-1)
    u,s,vh=np.linalg.svd(I,full_matrices=False);B=np.sqrt(s[:3])[:,None]*vh[:3];lighting=u[:,:3]*np.sqrt(s[:3])
    mask=images.mean(axis=0)>.05*images.mean(axis=0).max()

    def fields(b):
     a=np.linalg.norm(b,axis=0);n=(b/np.maximum(a,1e-12)).T.reshape(h,w,3)
     return a.reshape(h,w),n

    def normals(b,m=None,shape=(h,w)):
     a=np.linalg.norm(b,axis=0);n=(b/np.maximum(a,1e-12)).T.reshape(*shape,3)
     return (n+1)/2 if m is None else np.where(m[...,None],(n+1)/2,0)

    def albedo(b,m=mask,shape=(h,w)):
     a=np.linalg.norm(b,axis=0).reshape(shape)
     return np.where(m,a/max(np.percentile(a[m],99.5),1e-12),0)

    def slopes(b):
     a,n=fields(b);nz=n[...,2];den=np.where(nz>=0,1.,-1.)*np.maximum(abs(nz),.05)
     return np.where(mask,-n[...,0]/den,0),np.where(mask,-n[...,1]/den,0)

    def depth(b,method='poisson'):
     p,q=slopes(b);z=(integrate_poisson if method=='poisson' else integrate_frankot)(p.copy(),q.copy());z-=np.median(z[mask]);return z

    def montage(name,panels,ncols=3,size=None):
     rows=int(np.ceil(len(panels)/ncols));fig,axes=plt.subplots(rows,ncols,figsize=size or (ncols*3.2,rows*3.5),squeeze=False)
     for ax,(im,title) in zip(axes.ravel(),panels):
      # Orthographic outputs are heights: larger height is nearer the viewer.
      depth_panel = any(word in title.lower() for word in ['depth','poisson','frankot'])
      ax.imshow(np.clip(im,0,1) if im.ndim==3 else im,cmap='gray_r' if depth_panel else 'gray');ax.set_title(title);ax.axis('off')
     for ax in axes.ravel()[len(panels):]:ax.axis('off')
     fig.tight_layout();fig.savefig(OUT/name,dpi=180,bbox_inches='tight');plt.close(fig)

    def surfaces(name,bs,titles):
     fig=plt.figure(figsize=(4*len(bs),6));yy,xx=np.indices((h,w))
     for j,(b,title) in enumerate(zip(bs,titles)):
      z=depth(b)
      for row,az in enumerate([-65,30]):
       ax=fig.add_subplot(2,len(bs),row*len(bs)+j+1,projection='3d')
       ax.plot_surface(xx,yy,np.where(mask,z,np.nan),color='silver',rstride=3,cstride=3,linewidth=0,shade=True,lightsource=LightSource(azdeg=315,altdeg=35))
       ax.set_box_aspect((w,h,max(np.ptp(z[mask]),1)),zoom=1.45)
       ax.view_init(30,az);ax.invert_yaxis();ax.set_axis_off();ax.set_title(title if row==0 else 'Second view')
     fig.tight_layout();fig.savefig(OUT/name,dpi=160,bbox_inches='tight');plt.close(fig)

    def shoe_surface(b, foreground, filename, title):
     sh=foreground.shape
     n=(b/np.maximum(np.linalg.norm(b,axis=0),1e-12)).T.reshape(*sh,3)
     nz=n[...,2];den=np.where(nz>=0,1.,-1.)*np.maximum(abs(nz),.1)
     z=integrate_masked(-n[...,0]/den,-n[...,1]/den,foreground)
     y,x=np.indices(sh)
     fig=plt.figure(figsize=(8,4));ax=fig.add_subplot(111,projection='3d')
     ax.plot_surface(x,y,np.where(foreground,z,np.nan),color='silver',rstride=3,cstride=3,linewidth=0,lightsource=LightSource(315,35))
     ax.set_box_aspect((sh[1],sh[0],200));ax.set_proj_type('ortho')
     ax.view_init(elev=55,azim=-90);ax.invert_yaxis();ax.set_axis_off()
     fig.tight_layout();fig.savefig(OUT/filename,dpi=180,bbox_inches='tight');plt.close(fig)
     # Trim empty plotting margins so the surface is readable at report size.
     from PIL import Image, ImageOps
     image=Image.open(OUT/filename).convert('RGB');pixels=np.asarray(image)
     yy,xx=np.where(np.any(pixels<245,axis=2))
     image=ImageOps.expand(image.crop((xx.min(),yy.min(),xx.max()+1,yy.max()+1)),border=20,fill='white')
     image.save(OUT/filename)

    Q=np.array([[1,0,0],[0,1,0],[0,-1,1.]])
    bq=np.linalg.solve(Q.T,B);bv=B.copy();bv[2]*=-1;bqv=bq.copy();bqv[2]*=-1
    montage('face_inputs.png',[(images[i]/max(images.max(),1e-12),f'Input {i+1}') for i in range(7)],4)
    montage('face_svd.png',[(albedo(bv),'SVD albedo'),(normals(bv),'SVD normals'),(albedo(bqv),'Transformed albedo'),(normals(bqv),'Transformed normals')],4)
    sigmas=[1,2,4,8,12,20];integrable={sig:recover_integrable(images,sig)[0] for sig in sigmas}
    montage('sigma_sweep.png',[(normals(integrable[sig]),rf'$\sigma={sig}$') for sig in sigmas],3)
    montage('sigma_albedo_sweep.png',[(albedo(integrable[sig]),rf'$\sigma={sig}$') for sig in sigmas],3)
    b0=integrable[8]
    choices=[(.0,.0,.5),(0,0,1),(0,0,2),(.2,0,2),(0,.2,2)]
    manual=[]
    for mu,nu,lam in choices:
     g=np.array([[1.,0,0],[0,1,0],[mu,nu,lam]]);manual.append(np.linalg.solve(g.T,b0))
    chosen=manual[2]
    montage('face_gbr_normals.png',[(normals(b),rf'$(\mu,\nu,\lambda)={p}$') for b,p in zip(manual,choices)],5,size=(15,3.5))
    # Calibrated system in the supplied light coordinates, then reflect z for display.
    bcal=np.linalg.lstsq(load_sources(),I,rcond=None)[0];bcal_vis=bcal.copy();bcal_vis[2]*=-1
    montage('face_comparison.png',[(albedo(b0),'Integrable albedo'),(albedo(chosen),'Manual GBR albedo'),(albedo(bcal_vis),'Calibrated albedo'),(normals(b0),'Integrable normals'),(normals(chosen),'Manual GBR normals'),(normals(bcal_vis),'Calibrated normals'),(depth(chosen),'Manual GBR: Poisson depth'),(depth(chosen,'frankot'),'Manual GBR: Frankot depth'),(depth(bcal_vis),'Calibrated: Poisson depth')],3)
    ps=depth(chosen);fc=depth(chosen,'frankot')
    montage('integration_uncalibrated.png',[(depth(b0),'Integrability only: Poisson'),(ps,'Manual GBR: Poisson'),(fc,'Manual GBR: Frankot-Chellappa')],3,size=(10,4.5))
    surfaces('integration_surfaces.png',[b0,chosen],['Integrability only','Manual GBR (0, 0, 2)'])
    montage('integration.png',[(np.where(mask,ps,np.nan),'Manual GBR: Poisson'),(np.where(mask,fc,np.nan),'Manual GBR: Frankot-Chellappa'),(np.where(mask,depth(bcal_vis),np.nan),'Calibrated: Poisson')],3,size=(10,4.5))
    surfaces('face_surfaces.png',[b0,chosen,bcal_vis],['Integrability only','Manual GBR (0, 0, 2)','Known-light calibrated'])
    face_entropy=np.load(ROOT/'output/bonus3/reconstruction.npz')
    montage('face_entropy.png',[(albedo(face_entropy['B_before']),'Before: albedo'),(normals(face_entropy['B_before']),'Before: normals'),(np.where(mask,face_entropy['depth'],np.nan),'After: depth'),(albedo(face_entropy['B_after']),'After: albedo'),(normals(face_entropy['B_after']),'After: normals'),(plt.imread(ROOT/'output/bonus3/after_surface.png'),'After: surface')],3)
    for name in ['cup','shoes']:
     base=ROOT/'output/capture_tests'/name;d=np.load(base/'reconstruction.npz');m=d['mask'];shape=m.shape
     raw=cv2.imread(str(base/'linear_inputs/input_1.tif'),-1)[...,::-1].astype(float)/65535
     raw=np.clip(raw/max(np.percentile(raw,99.5),1e-12),0,1)**(1/2.2)
     b=d['B_before'];a=np.linalg.norm(b,axis=0).reshape(shape);n=(b/np.maximum(np.linalg.norm(b,axis=0),1e-12)).T.reshape(*shape,3)
     light=np.array([-.4,-.3,1.]);light/=np.linalg.norm(light)
     radiance=np.maximum(0,a*np.sum(n*light,axis=-1));radiance=np.where(m,radiance,0);radiance/=max(np.percentile(radiance[m],99.5),1e-12)
     save_image(base/'relighting.png',np.clip(radiance,0,1)**(1/2.2))
     before_depth = np.where(m,plt.imread(base/'before_depth.png'),np.nan)
     montage(name+'_baseline.png',[(raw,'Captured image 1'),(plt.imread(base/'segmentation_overlay.png'),'Object segmentation'),(albedo(b,m,shape),'Integrable albedo'),(normals(b,m,shape),'Integrable normals'),(before_depth,'Integrable depth'),(np.clip(radiance,0,1)**(1/2.2),'Novel-light Lambertian rendering')],3,size=(10,6.8))
     shutil.copy2(base/'perspective/overview.png',OUT/(name+'_perspective.png'))
     if name=='shoes':
      shoe_surface(b,m,'shoes_surface_baseline.png','Integrability only')
      shoe_surface(d['B_after'],m,'shoes_surface_entropy.png','Entropy GBR correction')
     else:
      shutil.copy2(base/'before_surface.png',OUT/'cup_surface_baseline.png')
     before_surface=plt.imread(OUT/(name+'_surface_baseline.png'))
     after_surface=plt.imread(OUT/'shoes_surface_entropy.png') if name=='shoes' else plt.imread(base/'after_surface.png')
     montage(name+'_entropy.png',[(plt.imread(base/'segmentation_overlay.png'),'Segmentation'),(normals(d['B_after'],m,shape),'Normals after entropy GBR'),(np.where(m,d['residual'],np.nan),'Rank-3 relative residual'),(before_depth,'Depth before GBR'),(np.where(m,d['depth'],np.nan),'Depth after GBR'),(after_surface,'Surface after GBR (arbitrary scale)')],3,size=(12,8))
     montage(name+'_surfaces.png',[(before_surface,'Integrability only'),(after_surface,'Entropy GBR correction')],2,size=(9,4))
    shutil.copy2(ROOT/'output/bonus3_2_focal/normal_comparison.png',OUT/'perspective_normals.png')
    shutil.copy2(ROOT/'output/bonus3_2_focal/selected/overview.png',OUT/'perspective_face.png')
    # Basis transformation must preserve the image fit exactly before visualization.
    err=float(np.max(abs(lighting@B-(lighting@Q.T)@bq)))
    report={'input_shape':[k,h,w],'singular_values':s.tolist(),'rank3_energy':float(np.sum(s[:3]**2)/np.sum(s**2)),'Q_image_fit_max_difference':err,
            'manual_gbr':{'mu':0,'nu':0,'lambda':2,'transform':'B_manual = inv(G).T @ B_integrable'},
            'calibrated_relative_residual':float(np.linalg.norm(I-load_sources()@bcal)/np.linalg.norm(I)),
            'poisson_frankot_mean_removed_depth_RMSE':float(np.sqrt(np.mean((ps[mask]-fc[mask])**2))),
            'novel_light':light.tolist()}
    (ROOT/'report'/'figure_metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

    from albedo_distributions import generate_distributions
    generate_distributions(ROOT)

    # Use the latest comparison outputs when the depth-improvement experiment exists.
    if (ROOT/"output/perspective_depth_diagnostics/improved/reconstruction.npz").exists():
     from update_perspective_report import update_figures
     update_figures(ROOT)


if __name__ == "__main__":
    main()
