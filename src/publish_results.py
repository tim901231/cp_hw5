"""Copy selected final images/metrics into the small, versioned results tree."""
from pathlib import Path
import shutil
import re
import json


def publish():
    root = Path(__file__).resolve().parents[1]
    figures = root / 'report' / 'figures'
    groups = {
        'problem1': ['face_svd.png', 'sigma_sweep.png', 'sigma_albedo_sweep.png',
                     'integration_uncalibrated.png', 'integration_surfaces.png', 'face_comparison.png'],
        'problem2/cup': ['cup_baseline.png', 'cup_surface_baseline.png'],
        'problem2/shoes': ['shoes_baseline.png', 'shoes_surface_baseline.png'],
        'bonus_entropy': ['face_entropy.png', 'cup_entropy.png', 'shoes_entropy.png',
                          'albedo_distributions.png', 'cup_surfaces.png', 'shoes_surfaces.png'],
        'bonus_perspective': ['perspective_normals.png', 'perspective_face.png',
                              'cup_perspective.png', 'shoes_perspective.png'],
    }
    for folder, names in groups.items():
        dest = root / 'results' / folder
        dest.mkdir(parents=True, exist_ok=True)
        for name in names:
            source = figures / name
            if not source.exists():
                raise FileNotFoundError(f'{source}: run the figures command first')
            shutil.copy2(source, dest / name)
    for source, target in [
        ('report/figure_metrics.json', 'results/problem1/metrics.json'),
        ('report/albedo_distribution_metrics.json', 'results/bonus_entropy/metrics.json'),
        ('output/perspective_depth_diagnostics/report_results/results.json', 'results/bonus_perspective/metrics.json'),
    ]:
        if (root / source).exists():
            shutil.copy2(root / source, root / target)
    # Keep the report's small parameter fragments in sync with this run.
    own = json.loads((root/'output/capture_tests/shoes/results.json').read_text())
    for label, value in zip(['mu', 'nu', 'lambda'], own['gbr_parameters']):
        (root/'report'/('boot_' + label + '.tex')).write_text(f'{value:.4f}')
    # Archive unused/generated extras rather than checking them into report/.
    tex = (root / 'report' / 'report.tex').read_text()
    used = set(re.findall(r'\\fig\{([^}]+)', tex) +
               re.findall(r'\\includegraphics\[[^]]*\]\{([^}]+)', tex))
    unused = root / 'output' / 'unused_report_figures'
    unused.mkdir(parents=True, exist_ok=True)
    for image in figures.glob('*.png'):
        if image.name not in used:
            shutil.move(str(image), str(unused / image.name))
    print('Selected results:', root / 'results')


if __name__ == '__main__':
    publish()
