# CP HW5: Photometric Stereo

15-663 Computational Photography, Homework 5. Uncalibrated and calibrated
photometric stereo, integrability and manual GBR transformations, reconstruction
of my cup and boot captures, and two bonus methods: albedo-entropy minimization
and perspective photometric stereo.

The report is [`report.pdf`](report.pdf). Editable LaTeX and the figures used by
the report are in [`report/`](report/). The layout follows
[CP_hw4](https://github.com/tim901231/CP_hw4): source code in `src/`, my inputs in
`data/own/`, selected images in `results/`, and generated arrays in ignored
`output/`.

## Setup

Tested with Python 3.8, NumPy 1.24, SciPy 1.10, Pillow 10.4 and Matplotlib 3.7.
Python 3.8–3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy the seven **linear** face TIFFs from the homework ZIP to:

```text
data/input_1.tif
...
data/input_7.tif
```

The course data and assignment handout are not committed, following HW4.
`src/cp_hw5.py` contains the supplied light vectors and integrators, so
`sources.mat` is not needed. My exact linear crops and segmentation masks are
committed in `data/own/`; running them does **not** require NEFs, `dcraw`, or
`exiftool`.

## Usage

Run from the repository root. Outputs are headless; no plot windows are opened.

```bash
python src/main.py all
```

This reconstructs both datasets, runs the bonus experiments, regenerates report
figures, and publishes the selected results. It does not rebuild the PDF.

| Command | What it does | Generated outputs |
|---|---|---|
| `python src/main.py p1` | SVD, linear ambiguity, integrability, manual GBR and calibrated face | `output/problem1/` |
| `python src/main.py p2` | Cup/boot segmentation, integrability, entropy GBR and initial perspective estimates | `output/capture_tests/` |
| `python src/main.py entropy` | Face entropy-GBR search | `output/bonus3/` |
| `python src/main.py perspective` | Initial face perspective estimate, reference-assisted parameter fitting, improved depth experiment | `output/bonus3_2/`, `output/bonus3_2_focal/`, `output/perspective_depth_diagnostics/improved/` |
| `python src/main.py figures` | Figures, albedo histograms, final weighted-depth panels and selected results | `report/figures/`, `results/` |
| `python src/main.py diagnostics` | Synthetic end-to-end check and real-data integration comparisons | `output/perspective_depth_diagnostics/` |

Run `p2`, `entropy` and `perspective` before `figures` in a fresh checkout, or use
`all`. The perspective reference-fitting stage needs the supplied face TIFFs.
Intermediate `.npz` arrays are intentionally not committed. `python src/main.py
-h` lists commands.

Individual entry points provide additional options:

```bash
python src/own_shapes.py --datasets cup       # cup only
python src/own_shapes.py --datasets shoes     # boot only
python src/entropy.py --sigma 8 --grid 9 --levels 5
python src/perspective.py --focal 1 --sigma 8
```

The scripts use paths relative to the project directory; `main.py` also sets the
working directory correctly for subprocesses. If your Matplotlib cache directory
is not writable, set `MPLCONFIGDIR` to a writable temporary directory.

## Build the report

The committed root PDF is the latest compact, PDF 1.4 compatibility version.
Building from the committed LaTeX and figures does not require the image inputs
or reconstruction arrays. A TeX installation and Ghostscript are needed:

```bash
cd report
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
cd ..
gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 \
  -dPDFSETTINGS=/printer -dDetectDuplicateImages=true \
  -dDownsampleColorImages=true -dColorImageResolution=180 \
  -dDownsampleGrayImages=true -dGrayImageResolution=180 \
  -sOutputFile=report.pdf report/report.pdf
```

The LaTeX source uses Latin Modern, microtype, subcaption, float, placeins,
booktabs, listings and hyperref (included with a standard TeX Live/MacTeX setup).

## Code

| File | Contents |
|---|---|
| `src/main.py` | Ordered command-line workflow |
| `src/photometric_stereo.py` | Part 1 numerical reconstruction and exports |
| `src/own_shapes.py` | Committed inputs, reproducible object masks, own-capture reconstructions; optional RAW preparation |
| `src/entropy.py` | SVD, integrability, entropy GBR, masked integration and image/surface export |
| `src/perspective.py` | Perspective integrability matrix, ambiguity recovery and original depth integration |
| `src/perspective_integration.py` | Incidence-weighted masked log-depth integration with Huber reweighting |
| `src/fit_perspective_reference.py` | First reference-assisted face parameter fit, retained as the comparison baseline |
| `src/perspective_depth_improvement.py` | Fixed smoothing/erosion grid and reference-assisted focal/normal rescaling for the final face result |
| `src/perspective_depth_diagnostics.py` | Synthetic test and integration comparisons; not required for the report |
| `src/report_figures.py` | Report figures, depth conventions and surface viewpoints |
| `src/albedo_distributions.py` | Before/after albedo histograms on the exact entropy-search samples |
| `src/update_perspective_report.py` | Final face/cup/boot perspective panels |
| `src/publish_results.py` | Copies selected images and metrics to `results/` |
| `src/cp_hw5.py`, `src/utils.py` | Course-provided integration, lighting and color helpers |

## My captures

| Inputs | Camera settings | Committed resolution (H × W) |
|---|---|---|
| `data/own/cup/input_1.tif` … `input_7.tif` | Nikon D3500, 55 mm, f/5.6, ISO 100, 4 s | 320 × 380 |
| `data/own/shoes/input_1.tif` … `input_7.tif` | Nikon D3500, 55 mm, f/5.6, ISO 100, 8 s | 440 × 810 |

These are the exact **linear 16-bit sRGB crops used in the report**, not JPEG
previews. The full RAW decode was resized to width 2000 before cropping. Fixed
unit white-balance multipliers and no automatic brightening preserve intensity
relationships across the seven lights. Each folder also contains `mask.png` and
`capture.json`. The committed masks were initialized with capture-specific
silhouette seeds and refined by GrabCut. The cup handle hole is excluded.

To prepare again from the original RAWs, put them in `data/cup/` or `data/shoes/`
and use `src/own_shapes.py --width <different-full-frame-width>`. This requires
`dcraw` and `exiftool` on PATH (on macOS: `brew install dcraw exiftool`). At the
default width 2000, the committed inputs and masks take precedence. Generated
TIFF caches with a different width are rejected; move their `linear_inputs/`
folder and `capture_metadata.json` out of `output/capture_tests/<object>/` before
switching width. New scenes also require updating the silhouette seeds and crop.

## Selected parameters and interpretation

| Stage | Settings |
|---|---|
| Face integrability | Gaussian derivative smoothing σ=8 |
| Manual face GBR | (μ, ν, λ)=(0, 0, 2), `inv(G).T @ B` |
| Own-shape integrability | σ=2; foreground-only SVD; masked integration |
| Face entropy | 256 bins, 9³ grid, 5 levels, all entropy inliers |
| Own-shape entropy | 256 bins, 7³ grid, 4 levels, 12000 fixed samples, seed 0 |
| Final perspective face | σ=8; 6-pixel erosion; effective f=1318.74; center (184, 310); 3 Huber iterations |
| Shoe surface display | elevation 55°, azimuth −90°; arbitrary vertical display scale |

The entropy implementation uses `G.T @ B`, so its parameters use a different
convention from the manual inverse-transpose GBR. Depth displays use black for
near and white for far.

**The final perspective face result is reference-assisted.** Known-light normals
on alternating 16-pixel training blocks select parameters and normal/focal
rescaling; held-out reference blocks report error. These are effective parameters,
not independently calibrated camera intrinsics. The weighted integrator extends
the paper's method. Cup/boot perspective normal estimates are retained while
their depth integration is updated. Cleaner appearance does not establish
correct depth, and the glossy cup violates the Lambertian model.

## Versioned files

- Commit code, this README, dependency constraints, the exact cropped own inputs
  and masks, selected final results, report LaTeX/used figures, and root `report.pdf`.
- Keep RAW captures, course inputs, exploratory sweeps, reconstruction arrays,
  intermediate TIFF caches, old PDF variants, ZIPs and TeX build files local.
- The original files were retained under ignored `output/` or their ignored RAW
  directories during cleanup; they were not deleted.

## Acknowledgement

The visualizations, numerical checks, report writing and repository organization
were prepared in collaboration with OpenAI Codex. Course-provided helpers are
retained with their original descriptions.
