"""Part 1: uncalibrated, integrable, manual-GBR and calibrated reconstruction."""
from pathlib import Path
import json
import numpy as np
from entropy import load_images, recover_integrable, export_reconstruction, save_image
from cp_hw5 import load_sources


def run():
    root = Path(__file__).resolve().parents[1]
    output = root / 'output' / 'problem1'
    output.mkdir(parents=True, exist_ok=True)
    images = load_images(root / 'data')
    count, h, w = images.shape
    intensity = images.reshape(count, -1)
    left, singular, right = np.linalg.svd(intensity, full_matrices=False)
    b = np.sqrt(singular[:3])[:, None] * right[:3]
    q = np.array([[1., 0, 0], [0, 1, 0], [0, -1, 1]])
    transformed = np.linalg.solve(q.T, b)
    # Display raw SVD normal maps without masking, as in the assignment.
    for name, value in [('svd', b), ('linear_ambiguity', transformed)]:
        shown = value.copy()
        shown[2] *= -1
        albedo = np.linalg.norm(shown, axis=0)
        normals = (shown / np.maximum(albedo, 1e-12)).T.reshape(h, w, 3)
        save_image(output / (name + '_albedo.png'), albedo.reshape(h, w) / albedo.max())
        save_image(output / (name + '_normal.png'), (normals + 1) / 2)
    integrable, residual = recover_integrable(images, 8.)
    foreground = images.mean(axis=0) > .05 * images.mean(axis=0).max()
    # Manual GBR uses the assignment's inverse-transpose convention.
    manual = np.linalg.solve(np.diag([1., 1., 2.]).T, integrable)
    calibrated = np.linalg.lstsq(load_sources(), intensity, rcond=None)[0]
    calibrated[2] *= -1
    for name, value in [('integrable', integrable), ('manual_gbr', manual), ('calibrated', calibrated)]:
        export_reconstruction(value, (h, w), foreground, output, name, .05)
    np.savez_compressed(output / 'reconstruction.npz', B=integrable, B_manual=manual,
                        B_calibrated=calibrated, foreground=foreground, residual=residual)
    (output / 'parameters.json').write_text(json.dumps({
        'sigma': 8, 'manual_gbr': [0, 0, 2], 'integration': 'Poisson',
        'rank3_energy_fraction': float(np.sum(singular[:3]**2)/np.sum(singular**2))
    }, indent=2) + '\n')
    print('Part 1 outputs:', output)


if __name__ == '__main__':
    run()
