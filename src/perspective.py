"""Perspective uncalibrated photometric stereo, Papadhimitri & Favaro (2013).

python "src/perspective.py" --focal 1 --sigma 8
Eq. 24: one row per pixel, nine columns. Eq. 23: Psi has psi_i as
COLUMNS, Q=Psi^{-T}, and recovered pseudo-normals are solve(Psi, B).
Eq. 27 integrates log camera distance, not orthographic depth slopes.

f=1 and the centered principal point are assumptions, not TIFF calibration.
Ordinary rank-3 SVD is used rather than the paper's robust preprocessing.
Reference-assisted experiments are separate: fit_perspective_reference.py.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from entropy import load_images, make_masks, save_image, integrate_masked
from cp_hw5 import integrate_poisson


def coordinates(shape, cx=None, cy=None):
    h, w = shape
    cx = (w-1)/2 if cx is None else cx
    cy = (h-1)/2 if cy is None else cy
    v, u = np.indices(shape, dtype=float)
    return u-cx, v-cy, cx, cy


def build_P(B, Bu, Bv, u, v, focal):
    """B/Bu/Bv: (3,M); u/v: M coordinates; result: (M,9)."""
    if focal <= 0:
        raise ValueError('Focal length must be positive.')
    pu = np.cross(Bu.T, B.T)
    pv = np.cross(Bv.T, B.T)
    return np.concatenate((pu, pv, -u.ravel()[:, None]/focal*pu
                           -v.ravel()[:, None]/focal*pv), axis=1)


def solve_ambiguity(P, B):
    _, singular, vh = np.linalg.svd(P, full_matrices=False)
    if len(singular) != 9 or singular[-2] <= singular[0]*1e-12:
        raise ValueError('Perspective constraints are degenerate (rank below 8).')
    # The null vector stacks psi_1, psi_2, psi_3; these are columns of Psi.
    psi = vh[-1].reshape(3, 3).T
    condition = np.linalg.cond(psi)
    if not np.isfinite(condition) or condition > 1e12:
        raise ValueError('Singular ambiguity matrix; inspect focal length/mask/data.')
    return np.linalg.solve(psi, B), psi, singular


def log_depth_gradients(normals, u, v, focal, ray_floor=1e-4):
    """Eq. 27 without separately dividing by Nz. Reject grazing viewing rays."""
    ray_norm = np.sqrt(u*u+v*v+focal*focal)
    denominator = u*normals[..., 0] + v*normals[..., 1] + focal*normals[..., 2]
    valid = np.isfinite(denominator) & (abs(denominator)/ray_norm > ray_floor)
    gu = np.divide(-normals[..., 0], denominator, out=np.zeros_like(u), where=valid)
    gv = np.divide(-normals[..., 1], denominator, out=np.zeros_like(v), where=valid)
    return gu, gv, valid, denominator


def run(args):
    images = load_images(args.data)
    count, h, w = images.shape
    u, v, cx, cy = coordinates((h, w), args.cx, args.cy)
    intensity = images.reshape(count, -1)
    masked_domain = getattr(args, 'masked_domain', False)
    if masked_domain:
        if args.mask is None:
            raise ValueError('--masked-domain requires --mask')
        object_mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
        if object_mask is None or object_mask.shape != (h,w):
            raise ValueError('Mask must match input image dimensions')
        intensity = intensity.copy()
        intensity[:, object_mask.ravel() == 0] = 0
    left, s, right = np.linalg.svd(intensity, full_matrices=False)
    B = np.sqrt(s[:3])[:, None]*right[:3]
    lighting = left[:, :3]*np.sqrt(s[:3])
    residual = np.linalg.norm(intensity-lighting@B, axis=0)
    residual /= np.maximum(np.linalg.norm(intensity, axis=0), 1e-12)
    foreground, inliers = make_masks(images, residual.reshape(h, w), args.mask)
    if masked_domain:
        radius = max(1,int(np.ceil(3*args.sigma)))
        inliers &= cv2.erode(foreground.astype(np.uint8), np.ones((2*radius+1,)*2,np.uint8)) > 0
    smooth = B.T.reshape(h, w, 3)
    if args.sigma > 0:
        smooth = cv2.GaussianBlur(smooth, (0, 0), args.sigma)
    Bu = np.gradient(smooth, axis=1).reshape(-1, 3).T
    Bv = np.gradient(smooth, axis=0).reshape(-1, 3).T
    P = build_P(B, Bu, Bv, u, v, args.focal)
    recovered, psi, singular = solve_ambiguity(P[inliers.ravel()], B)
    albedo = np.linalg.norm(recovered, axis=0).reshape(h, w)
    normals = (recovered/np.maximum(albedo.ravel(), 1e-15)).T.reshape(h, w, 3)
    # Resolve only the global SVD sign, never flip an individual normal channel.
    ray_dot = u*normals[..., 0]+v*normals[..., 1]+args.focal*normals[..., 2]
    if np.median(ray_dot[inliers]) < 0:
        recovered *= -1
        normals *= -1
        psi *= -1
    gu, gv, valid, denominator = log_depth_gradients(normals, u, v, args.focal, args.ray_floor)
    valid &= foreground
    excluded_depth_pixels = int((foreground & ~valid).sum())
    # Rectangular Poisson integration with zero slopes at excluded pixels.
    # This is a numerical approximation, not a masked-domain Poisson solver.
    g = (integrate_masked(gu, gv, valid) if masked_domain else
         integrate_poisson(np.where(valid, gu, 0), np.where(valid, gv, 0)))
    if masked_domain:
        foreground = valid
    g -= np.median(g[foreground])
    if np.max(abs(g[foreground])) > 500:
        raise ValueError('Log depth diverged; check focal length, mask and viewing-ray singularities.')
    distance = np.exp(g)  # median distance fixed to 1: arbitrary global scale
    depth = distance-args.focal
    X, Y = u*distance/args.focal, v*distance/args.focal
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    save_image(out/'albedo.png', np.where(foreground, albedo/max(albedo[foreground].max(),1e-12),0))
    save_image(out/'normal.png', np.where(foreground[..., None],(normals+1)/2,0))
    lo, hi = np.percentile(depth[foreground], [1,99])
    save_image(out/'depth.png', np.where(foreground,(depth-lo)/max(hi-lo,1e-12),0))
    for name, mask in [('foreground_mask',foreground),('integrability_mask',inliers),('depth_mask',valid)]:
        save_image(out/f'{name}.png',mask.astype(float))
    fig, axes = plt.subplots(1,3,figsize=(12,5))
    axes[0].imshow(np.where(foreground,albedo,0),cmap='gray');axes[0].set_title('Albedo')
    axes[1].imshow(np.where(foreground[...,None],(normals+1)/2,0));axes[1].set_title('Normals')
    axes[2].imshow(np.where(foreground,depth,np.nan),cmap='gray',vmin=lo,vmax=hi);axes[2].set_title('Depth (1-99% display range)')
    for ax in axes: ax.axis('off')
    fig.tight_layout();fig.savefig(out/'overview.png',dpi=150);plt.close(fig)
    fig=plt.figure(figsize=(8,7));ax=fig.add_subplot(111,projection='3d')
    ax.plot_surface(X,Y,np.where(valid,-distance,np.nan),cmap='gray',rstride=4,cstride=4)
    ax.invert_yaxis();ax.set_axis_off()
    fig.savefig(out/'surface.png',dpi=150,bbox_inches='tight');plt.close(fig)
    np.savez_compressed(out/'reconstruction.npz',B=B,B_recovered=recovered,P=P,Psi=psi,
                        Q=np.linalg.inv(psi).T,singular_values=singular,normals=normals,
                        albedo=albedo,log_distance=g,distance=distance,depth=depth,
                        X=X,Y=Y,u=u,v=v,foreground=foreground,inliers=inliers,depth_valid=valid)
    report={'focal_pixels':args.focal,'cx':cx,'cy':cy,'sigma':args.sigma,
            'P_shape':list(P.shape),'fit_pixels':int(inliers.sum()),
            'singular_values':singular.tolist(),'nullspace_gap':float(singular[-2]/max(singular[-1],1e-30)),
            'Psi_condition':float(np.linalg.cond(psi)),
            'excluded_depth_pixels':excluded_depth_pixels,
            'log_distance_range': [float(g[foreground].min()),float(g[foreground].max())],
            'depth_display_percentiles':[1,99], 'median_camera_distance':1.0,
            'masked_domain':masked_domain,
            'limitations':['Intrinsics assumed unless supplied; no calibrated scale correction.',
                          'Ordinary SVD and derivative smoothing, not paper robust low-rank preprocessing.',
                          ('Masked-domain integration with one arbitrary offset per component.' if masked_domain else
                           'Excluded depth slopes set to zero for rectangular Poisson integration.'),
                          'Near-orthographic data may be poorly conditioned for this method.']}
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));print(f'Saved to {out}')


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=root/'data')
    parser.add_argument('--output',type=Path,default=root/'output'/'bonus3_2')
    parser.add_argument('--mask',type=Path)
    parser.add_argument('--masked-domain', action='store_true', help='Exclude background from SVD and integrate only object edges')
    parser.add_argument('--focal',type=float,default=1.)
    parser.add_argument('--cx',type=float)
    parser.add_argument('--cy',type=float)
    parser.add_argument('--sigma',type=float,default=8.)
    parser.add_argument('--ray-floor',type=float,default=1e-4)
    args=parser.parse_args()
    if not np.isfinite([args.focal,args.sigma,args.ray_floor]).all() or args.focal<=0 or args.sigma<0 or not 0<args.ray_floor<1:
        parser.error('Require focal>0, sigma>=0, and 0<ray-floor<1.')
    run(args)


if __name__=='__main__':
    main()
