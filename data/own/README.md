# Own photometric-stereo inputs

`cup/` and `shoes/` each contain seven linear 16-bit sRGB TIFFs, the fixed
segmentation mask used by the report, and capture metadata. These crops exactly
match the report resolution. Use `python src/main.py p2` from the repository root;
RAW decoders are not needed for these committed inputs.

The TIFFs were decoded with `dcraw -4 -h -r 1 1 1 1 -o 1`, resized to full-frame
width 2000, and cropped. Do not apply inverse gamma or per-frame autoexposure.
`mask.png` is nonzero on the object; the cup handle hole is background.

The original NEFs remain local and ignored. See the root README for full capture
settings, optional RAW preparation, reconstruction commands and limitations.
