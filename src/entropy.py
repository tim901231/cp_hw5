"""Alldrin et al. (CVPR 2007), Sections 4.1-4.3: albedo entropy GBR search.

Run: python src/entropy.py --data data --output output/bonus3
Optional: --mask object_mask.png (nonzero pixels are foreground).

B uses columns (3 x pixels), so the paper's recovery B_hat @ G becomes
G.T @ B here, NOT inv(G).T @ B. Histograms have candidate-dependent support.
This implementation uses ordinary SVD plus a fixed entropy inlier mask;
it does not implement the paper's missing-data/visibility-weighted SVD.
"""
import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

from utils import lRGB2XYZ
from cp_hw5 import integrate_poisson


def load_images(directory):
    paths = sorted(Path(directory).glob('input_*.tif'))
    if len(paths) < 3:
        raise ValueError('Need at least three input_*.tif images.')
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f'Expected RGB TIFF: {path}')
        scale = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else 1.0
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64) / scale
        images.append(lRGB2XYZ(rgb)[..., 1])
    return np.stack(images)


def recover_integrable(images, sigma, mask=None):
    """Assignment's SVD + Yuille/Snow integrability, prior to GBR recovery."""
    count, h, w = images.shape
    intensity = images.reshape(count, -1).copy()
    if mask is not None:
        intensity[:, ~mask.ravel()] = 0
    u, s, vh = np.linalg.svd(intensity, full_matrices=False)
    B = np.sqrt(s[:3])[:, None] * vh[:3]
    lighting = u[:, :3] * np.sqrt(s[:3])
    residual = np.linalg.norm(intensity - lighting @ B, axis=0)
    residual /= np.maximum(np.linalg.norm(intensity, axis=0), 1e-12)
    smooth = cv2.GaussianBlur(B.T.reshape(h, w, 3), (0, 0), sigma)
    dy = np.gradient(smooth, axis=0).reshape(-1, 3).T
    dx = np.gradient(smooth, axis=1).reshape(-1, 3).T
    constraints = np.stack([
        B[0]*dx[1] - B[1]*dx[0],
        B[0]*dx[2] - B[2]*dx[0],
        B[1]*dx[2] - B[2]*dx[1],
        -B[0]*dy[1] + B[1]*dy[0],
        -B[0]*dy[2] + B[2]*dy[0],
        -B[1]*dy[2] + B[2]*dy[1],
    ], axis=1)
    if mask is not None:
        # Avoid derivatives across the silhouette and large non-Lambertian residuals.
        inner = cv2.erode(mask.astype(np.uint8), np.ones((2*int(np.ceil(3*sigma))+1,)*2, np.uint8)) > 0
        inner &= residual.reshape(h, w) <= np.quantile(residual[mask.ravel()], .9)
        if inner.sum() < 100:
            raise ValueError('Too few interior pixels for integrability.')
        constraints = constraints[inner.ravel()]
    _, _, vh = np.linalg.svd(constraints, full_matrices=False)
    x = vh[-1]
    delta = np.array([[-x[2], x[5], 1], [x[1], -x[4], 0], [-x[0], x[3], 0]])
    if np.linalg.cond(delta) > 1e10:
        raise ValueError('Integrability transform is ill-conditioned; check images/sigma.')
    B = np.linalg.solve(delta, B)
    # Match main.py: resolve the concave/convex branch AFTER integrability.
    B[2] *= -1
    return B, residual.reshape(h, w)


def make_masks(images, residual, mask_path=None):
    """Fixed (not candidate-dependent) masks for object display and entropy."""
    mean = images.mean(axis=0)
    if mask_path:
        foreground = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if foreground is None or foreground.shape != mean.shape:
            raise ValueError('Mask must be readable and match the image dimensions.')
        foreground = foreground > 0
    else:
        # A conservative intensity mask, not semantic object segmentation.
        foreground = mean > 0.05 * mean.max()
    if foreground.sum() < 100:
        raise ValueError('Foreground mask contains fewer than 100 pixels.')
    edge = np.zeros_like(mean)
    for image in images:
        gy, gx = np.gradient(image)
        edge = np.maximum(edge, np.hypot(gx, gy))
    inner = cv2.erode(foreground.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    inliers = (inner & (residual <= np.quantile(residual[foreground], .90))
               & (edge <= np.quantile(edge[foreground], .90)))
    if inliers.sum() < 100:
        raise ValueError('Too few entropy inliers after filtering.')
    return foreground, inliers


def gbr_matrix(parameters):
    mu, nu, lam = parameters
    return np.array([[1., 0., 0.], [0., 1., 0.], [mu, nu, lam]])


def histogram_entropy(albedo, bins=256):
    """Paper Eq. 15: discrete histogram entropy, no log(bin_width) term."""
    lo, hi = float(albedo.min()), float(albedo.max())
    if not np.isfinite([lo, hi]).all():
        return float('inf')
    if hi - lo <= 1e-12 * max(abs(lo), abs(hi), 1e-30):
        return 0.0
    counts = np.histogram(albedo, bins=bins, range=(lo, hi))[0]
    p = counts[counts > 0] / albedo.size
    return float(-np.sum(p * np.log(p)))


def entropy_search(B, bins=256, grid=9, levels=5):
    """Uniform coarse-to-fine grids, retaining the best candidate across levels."""
    lower = np.array([-5., -5., .01])
    upper = np.array([5., 5., 5.])
    lo, hi = lower.copy(), upper.copy()
    best = np.array([0., 0., 1.])

    def cost(parameters):
        mu, nu, lam = parameters
        albedo = np.sqrt((B[0] + mu*B[2])**2 + (B[1] + nu*B[2])**2
                         + (lam*B[2])**2)
        return histogram_entropy(albedo, bins)

    best_cost = cost(best)
    history = []
    for level in range(levels):
        step = (hi - lo) / (grid - 1)
        for candidate in itertools.product(*(np.linspace(a, b, grid) for a, b in zip(lo, hi))):
            score = cost(candidate)
            if score < best_cost:
                best_cost, best = score, np.array(candidate)
        history.append({'level': level, 'parameters': best.tolist(),
                        'entropy': best_cost, 'spacing': step.tolist()})
        print(f'Level {level+1}: mu, nu, lambda={best}, entropy={best_cost:.6f}', flush=True)
        lo, hi = np.maximum(lower, best-step), np.minimum(upper, best+step)
    return best, best_cost, history


def save_image(path, image):
    # Matplotlib/PIL use RGB; OpenCV writing would require RGB -> BGR.
    from PIL import Image
    Image.fromarray(np.round(np.clip(image, 0, 1)*255).astype(np.uint8)).save(path)


def integrate_masked(zx, zy, mask):
    """Least-squares gradient integration using only edges inside each component.

    One depth value per connected component is anchored to fix additive offsets.
    Background and holes impose no artificial zero-slope boundary constraints.
    """
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import spsolve
    from scipy.ndimage import label
    ids = np.full(mask.shape, -1, dtype=int)
    ids[mask] = np.arange(mask.sum())
    a, b, gradients = [], [], []
    for left, right, grad in [(ids[:, :-1], ids[:, 1:], .5*(zx[:, :-1]+zx[:, 1:])),
                              (ids[:-1], ids[1:], .5*(zy[:-1]+zy[1:]))]:
        valid = (left >= 0) & (right >= 0)
        a.append(left[valid]); b.append(right[valid]); gradients.append(grad[valid])
    a, b, g = np.concatenate(a), np.concatenate(b), np.concatenate(gradients)
    n = int(mask.sum())
    matrix = coo_matrix((np.concatenate([np.ones(len(a)), np.ones(len(a)), -np.ones(len(a)), -np.ones(len(a))]),
                         (np.concatenate([a,b,a,b]),np.concatenate([a,b,b,a]))), shape=(n,n)).tocsr()
    rhs = np.bincount(b, weights=g, minlength=n)-np.bincount(a, weights=g, minlength=n)
    components, count = label(mask)
    anchor = np.zeros(n)
    for component in range(1,count+1):
        anchor[ids[components == component][0]] = 1
    solution = spsolve(matrix + diags(anchor), rhs)
    depth = np.full(mask.shape, np.nan)
    depth[mask] = solution - np.median(solution)
    return depth


def export_reconstruction(B, shape, foreground, output, prefix, nz_floor, masked=False):
    h, w = shape
    albedo = np.linalg.norm(B, axis=0)
    normals = (B / np.maximum(albedo, 1e-12)).T.reshape(h, w, 3)
    albedo = albedo.reshape(h, w)
    save_image(output / f'{prefix}_albedo.png',
               np.where(foreground, albedo / max(albedo[foreground].max(), 1e-12), 0))
    save_image(output / f'{prefix}_normal.png', np.where(foreground[..., None], (normals+1)/2, 0))
    nz = normals[..., 2]
    denominator = np.where(nz >= 0, 1., -1.) * np.maximum(abs(nz), nz_floor)
    # z=f(x,y): n is proportional to (-zx, -zy, 1).
    zx = np.where(foreground, -normals[..., 0]/denominator, 0)
    zy = np.where(foreground, -normals[..., 1]/denominator, 0)
    depth = integrate_masked(zx, zy, foreground) if masked else integrate_poisson(zx.copy(), zy.copy())
    # Full rectangular Poisson solve with zero background slopes, not a masked solver.
    lo, hi = depth[foreground].min(), depth[foreground].max()
    display = (depth-lo) / max(hi-lo, 1e-12)
    save_image(output / f'{prefix}_depth.png', np.where(foreground, display, 0))
    y, x = np.indices(shape)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    # Display the height z used by (-zx, -zy, 1), without an extra z flip.
    surface = np.where(foreground, depth, np.nan)
    colors = LightSource().shade(np.nan_to_num(depth), cmap=plt.cm.gray)
    ax.plot_surface(x, y, surface, facecolors=colors, rstride=4, cstride=4)
    ax.invert_yaxis()
    ax.set_axis_off()
    fig.savefig(output / f'{prefix}_surface.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    return albedo, normals, depth


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=root/'data')
    parser.add_argument('--output', type=Path, default=root/'output'/'bonus3')
    parser.add_argument('--mask', type=Path)
    parser.add_argument('--sigma', type=float, default=8.)
    parser.add_argument('--bins', type=int, default=256)
    parser.add_argument('--grid', type=int, default=9)
    parser.add_argument('--levels', type=int, default=5)
    parser.add_argument('--samples', type=int, default=0, help='0 uses all inliers; otherwise fixed random subset')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nz-floor', type=float, default=.05)
    args = parser.parse_args()
    if args.sigma <= 0 or args.grid < 3 or args.levels < 1 or args.bins < 2 or args.samples < 0 or args.nz_floor <= 0:
        parser.error('Invalid sigma/grid/levels/bins/samples/nz-floor.')
    images = load_images(args.data)
    B, residual = recover_integrable(images, args.sigma)
    foreground, inliers = make_masks(images, residual, args.mask)
    indices = np.flatnonzero(inliers)
    if args.samples and len(indices) > args.samples:
        indices = np.random.default_rng(args.seed).choice(indices, args.samples, replace=False)
    if len(indices) < 100:
        parser.error('Use at least 100 entropy samples.')
    sample = B[:, indices]
    initial = histogram_entropy(np.linalg.norm(sample, axis=0), args.bins)
    print(f'Entropy pixels: {len(indices)}; initial entropy: {initial:.6f}', flush=True)
    best, score, history = entropy_search(sample, args.bins, args.grid, args.levels)
    G = gbr_matrix(best)
    corrected = G.T @ B
    args.output.mkdir(parents=True, exist_ok=True)
    before = export_reconstruction(B, images.shape[1:], foreground, args.output, 'before', args.nz_floor)
    after = export_reconstruction(corrected, images.shape[1:], foreground, args.output, 'after', args.nz_floor)
    save_image(args.output/'foreground_mask.png', foreground.astype(float))
    save_image(args.output/'entropy_mask.png', inliers.astype(float))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, values, title in zip(axes, [before[0], after[0]], ['Before', 'After']):
        ax.hist(values.ravel()[indices], bins=args.bins)
        ax.set(title=title, xlabel='Albedo', ylabel='Pixel count')
    fig.tight_layout()
    fig.savefig(args.output/'albedo_histograms.png', dpi=150)
    plt.close(fig)
    np.savez_compressed(args.output/'reconstruction.npz', B_before=B, B_after=corrected, G=G,
                        albedo=after[0], normals=after[1], depth=after[2],
                        foreground=foreground, inliers=inliers, sample_indices=indices)
    report = {'mu': float(best[0]), 'nu': float(best[1]), 'lambda': float(best[2]),
              'G': G.tolist(), 'transform': 'B_after = G.T @ B_before',
              'entropy_before': initial, 'entropy_after': score, 'history': history,
              'samples': len(indices), 'bins': args.bins, 'sigma': args.sigma,
              'nz_floor': args.nz_floor, 'mask': str(args.mask) if args.mask else 'automatic intensity threshold',
              'limitations': ['Ordinary SVD, not visibility-weighted missing-data SVD.',
                              'Entropy does not resolve the concave/convex branch.',
                              'Depth uses clamped slopes and a rectangular Poisson solve.']}
    if np.any(np.isclose(best, [-5, -5, .01])) or np.any(np.isclose(best, [5, 5, 5])):
        report['boundary_warning'] = 'Optimum is at a search bound; inspect mask and expand bounds if appropriate.'
        print(report['boundary_warning'])
    (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()
