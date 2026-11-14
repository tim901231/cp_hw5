"""Command-line entry point. Run `python src/main.py -h` for the workflow."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def script(name, *arguments):
    env = os.environ.copy()
    env.setdefault('MPLBACKEND', 'Agg')
    env.setdefault('OPENBLAS_NUM_THREADS', '1')
    env.setdefault('OMP_NUM_THREADS', '1')
    subprocess.run([sys.executable, str(ROOT / 'src' / name), *arguments],
                   cwd=ROOT, env=env, check=True)


def run(command):
    if command == 'p1':
        script('photometric_stereo.py')
    elif command == 'p2':
        script('own_shapes.py')
    elif command == 'entropy':
        script('entropy.py')
    elif command == 'perspective':
        script('perspective.py', '--focal', '1', '--sigma', '8')
        script('fit_perspective_reference.py')
        script('perspective_depth_improvement.py')
    elif command == 'figures':
        script('report_figures.py')
        script('publish_results.py')
    elif command == 'diagnostics':
        script('perspective_depth_diagnostics.py')
    elif command == 'all':
        for step in ['p1', 'p2', 'entropy', 'perspective', 'figures']:
            run(step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['p1', 'p2', 'entropy', 'perspective',
                                          'figures', 'diagnostics', 'all'])
    args = parser.parse_args()
    try:
        run(args.command)
    except subprocess.CalledProcessError as error:
        parser.exit(error.returncode, '\nPipeline step failed; see the error above.\n')


if __name__ == '__main__':
    main()
