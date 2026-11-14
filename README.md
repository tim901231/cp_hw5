# Homework 5: Photometric Stereo

The report is `report/report.pdf`; its editable LaTeX source is
`report/report.tex`. It follows the layout used in Homework 4.

## Environment

Python with NumPy, SciPy, OpenCV, Pillow and Matplotlib. RAW processing also
requires `dcraw` and `exiftool` on PATH. Building the PDF requires a LaTeX
installation with Latin Modern, microtype, subcaption, placeins and listings.
The tested local Python environment is:
`/Users/tsaiiast/miniconda3/envs/cp_hw1/bin/python`.
Set `MPLCONFIGDIR=/tmp/assgn5-mpl` if the default Matplotlib cache is unwritable.

## Data

- Supplied face: `data/input_1.tif` through `data/input_7.tif`.
- Cup and boot: seven NEFs each in `data/cup` and `data/shoes`.
- Generated capture outputs: `data/capture_tests/{cup,shoes}`.
- Face entropy experiment: `data/bonus3`.
- Original perspective face: `data/bonus3_2`.
- Reference-assisted diagnostic: `data/bonus3_2_focal`.

The report-source ZIP contains figures and code, not the original capture data.
Run numerical reproduction from the assignment directory with the data present.

## Reproduce

```sh
# Run in the assignment root.
python src/bonus3.py --data data --output data/bonus3
python src/bonus_3.2.py --data data --output data/bonus3_2 --focal 1 --sigma 8
python src/fit_perspective_reference.py
python src/test_captures.py
python src/build_report_figures.py
cd report
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

`fit_perspective_reference.py` uses known-light normals to select effective
intrinsics. It is a reference-assisted diagnostic, not a fully uncalibrated
method. Its holdout protocol and limitations are stated in the report.

`test_captures.py` reuses generated linear TIFFs. If the input NEFs change,
remove only the corresponding generated `linear_inputs` folder before rerunning.
The silhouette seeds are specific to these capture positions. See
`data/capture_tests/README.md` for numerical details and limitations.

`build_report_figures.py` adds a face smoothing sweep, non-diagonal basis-change
check, manual GBR experiments, calibrated reconstruction, integration comparison,
novel-light object renderings, and the figure assets used in the report. It also
writes `report/figure_metrics.json`.

The original interactive script is `src/main.py`. The headless scripts above
are the reproducible entry points used for this report.

## Additional perspective-depth checks

```sh
python src/perspective_depth_diagnostics.py
python src/perspective_ambiguity_sweep.py
python src/perspective_depth_improvement.py
```

These keep the original outputs and write to `data/perspective_depth_diagnostics`.
The improved result uses training known-light normal references for focal/normal
rescaling and parameter selection; validation normal references are withheld.
This is not a fully uncalibrated reconstruction, and no depth ground truth is
available. The weighted solver is an experimental extension to the paper.
