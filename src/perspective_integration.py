"""Confidence-weighted masked integration of perspective log-depth gradients."""
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.ndimage import label

def weighted_integrate(gu,gv,mask,confidence,iterations=1):
    ids=np.full(mask.shape,-1,int);ids[mask]=np.arange(mask.sum());a=[];b=[];targets=[];weights=[]
    for left,right,g,c in [(ids[:,:-1],ids[:,1:],.5*(gu[:,:-1]+gu[:,1:]),np.minimum(confidence[:,:-1],confidence[:,1:])),(ids[:-1],ids[1:],.5*(gv[:-1]+gv[1:]),np.minimum(confidence[:-1],confidence[1:]))]:
        good=(left>=0)&(right>=0)
        a.append(left[good]);b.append(right[good]);targets.append(g[good]);weights.append(c[good])
    a,b,t,w=[np.concatenate(x) for x in [a,b,targets,weights]]
    w=np.maximum(w,1e-10);n=int(mask.sum());component,nc=label(mask);anchor=np.zeros(n)
    for j in range(1,nc+1):anchor[ids[component==j][0]]=1
    robust=np.ones(len(w));history=[]
    for _ in range(iterations):
        ww=w*robust
        matrix=coo_matrix((np.concatenate([ww,ww,-ww,-ww]),(np.concatenate([a,b,a,b]),np.concatenate([a,b,b,a]))),shape=(n,n)).tocsr()+diags(anchor)
        rhs=np.bincount(b,weights=ww*t,minlength=n)-np.bincount(a,weights=ww*t,minlength=n)
        z=spsolve(matrix,rhs);error=z[b]-z[a]-t
        # Robustness in the original confidence-scaled gradient residual.
        residual=np.sqrt(w)*error;scale=max(1.4826*np.median(abs(residual-np.median(residual))),1e-8)
        robust=np.minimum(1.,1.345*scale/np.maximum(abs(residual),1e-20))
        history.append(float(np.sqrt(np.mean(residual**2))))
    output=np.full(mask.shape,np.nan);output[mask]=z-np.median(z)
    return output,history


def from_normals(normals,u,v,f,mask,iterations=1):
    ray=np.sqrt(u*u+v*v+f*f);den=u*normals[...,0]+v*normals[...,1]+f*normals[...,2]
    # Bounded division only for floating-point safety; tiny weights reject grazing data.
    floor=1e-8*ray;safe=np.where(den>=0,1.,-1.)*np.maximum(abs(den),floor)
    gu=-normals[...,0]/safe;gv=-normals[...,1]/safe
    confidence=(abs(den)/ray)**2
    return weighted_integrate(gu,gv,mask,confidence,iterations)


