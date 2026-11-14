"""Segment and evaluate the local cup/shoes RAW photometric-stereo captures.

Run: python src/own_shapes.py
Uses the committed linear TIFFs by default. Only optional RAW preparation
requires dcraw and exiftool; Python dependencies are in requirements.txt.
RAWs are decoded linearly with fixed unit white balance, never auto-brightened.
Segmentation uses capture-specific silhouette seeds refined by GrabCut on a
multi-light composite. These seeds are not a general semantic segmenter.
"""
import argparse
import json
import subprocess
import sys
import shutil
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from entropy import (load_images, recover_integrable, make_masks, entropy_search,
                    gbr_matrix, export_reconstruction, save_image)

ROOT = Path(__file__).resolve().parents[1]
# Coordinates are in a 1000-pixel-wide full-frame preview, y has the same scale.
SEEDS = {
 'cup': {'roi': [235, 425, 425, 585], 'polygon': [[289,447],[313,441],[373,441],[406,446],[407,552],[397,565],[375,572],[315,572],[293,564],[289,538],[269,535],[254,524],[249,504],[249,480],[255,465],[270,458],[288,458]],
         'holes': [[[272,474],[286,473],[287,519],[271,518],[262,508],[260,493],[264,481]]]},
 'shoes': {'roi': [205, 315, 610, 535], 'polygon': [[245,348],[292,340],[330,347],[348,330],[365,328],[378,342],[382,360],[399,380],[422,399],[441,416],[466,428],[499,435],[543,440],[576,446],[589,455],[588,481],[578,500],[555,511],[494,519],[414,523],[347,519],[281,515],[242,510],[226,500],[220,485],[225,466],[237,450],[240,405]], 'holes': []}
}


def prepare(name, width):
    directory = ROOT/'data'/name
    output = ROOT/'output'/'capture_tests'/name
    prepared = output/'linear_inputs'
    prepared.mkdir(parents=True, exist_ok=True)
    previous = output/'capture_metadata.json'
    if previous.exists() and json.loads(previous.read_text())['full_resized_width'] != width:
        raise ValueError('Cached TIFFs use another width; choose the original width or remove linear_inputs first.')
    committed = ROOT/'data'/'own'/name
    own_input = (committed/'capture.json').exists() and width == 2000
    if own_input:
        settings = json.loads((committed/'capture.json').read_text())
        meta = settings['raw_settings']
        for source in sorted(committed.glob('input_*.tif')):
            shutil.copy2(source, prepared/source.name)
    else:
        if not list(directory.glob('*.NEF')):
            raise FileNotFoundError(f'Need data/own/{name} at width 2000, or original NEFs in {directory}')
        meta = json.loads(subprocess.check_output(['exiftool','-json','-ExposureTime','-ISO','-FNumber','-FocalLength','-n',str(directory)]))
    for key in ['ExposureTime','ISO','FNumber','FocalLength']:
        if len({item.get(key) for item in meta}) != 1:
            raise ValueError('Capture settings differ: '+key)
    spec = SEEDS[name]
    for i, raw in enumerate(sorted(directory.glob('*.NEF')), 1):
        dest = prepared/f'input_{i}.tif'
        if dest.exists():
            continue
        print('Decoding', raw.name, name, flush=True)
        decoded = subprocess.check_output(['dcraw','-c','-4','-h','-r','1','1','1','1','-o','1',str(raw)])
        image = cv2.imdecode(np.frombuffer(decoded, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError('dcraw decode failed')
        image = cv2.resize(image, (width, round(image.shape[0]*width/image.shape[1])), interpolation=cv2.INTER_AREA)
        x0,y0,x1,y1 = np.round(np.array(spec['roi'])*width/1000).astype(int)
        cv2.imwrite(str(dest), image[y0:y1,x0:x1])
    paths = sorted(prepared.glob('input_*.tif'))
    if len(paths) != 7:
        raise ValueError('Expected seven captures')
    stack = np.stack([cv2.imread(str(p),cv2.IMREAD_UNCHANGED).astype(np.float32)/65535 for p in paths])
    # One shared display scale across the seven frames, used ONLY for segmentation.
    composite = np.max(stack,axis=0)
    scale = np.percentile(composite,99.5)
    display = np.uint8(np.clip(composite/max(scale,1e-8),0,1)**(1/2.2)*255)
    h,w = display.shape[:2]
    polygon = np.round((np.array(spec['polygon'])-np.array(spec['roi'][:2]))*width/1000).astype(np.int32)
    seed = np.zeros((h,w),np.uint8)
    cv2.fillPoly(seed,[polygon],1)
    for hole in spec['holes']:
        points = np.round((np.array(hole)-np.array(spec['roi'][:2]))*width/1000).astype(np.int32)
        cv2.fillPoly(seed,[points],0)
    band = max(3,round(width/100))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*band+1,2*band+1))
    labels = np.full((h,w),cv2.GC_BGD,np.uint8)
    labels[cv2.dilate(seed,kernel)>0] = cv2.GC_PR_BGD
    labels[seed>0] = cv2.GC_PR_FGD
    labels[cv2.erode(seed,kernel)>0] = cv2.GC_FGD
    cv2.setRNGSeed(0)
    cv2.grabCut(display,labels,None,np.zeros((1,65)),np.zeros((1,65)),7,cv2.GC_INIT_WITH_MASK)
    mask = (labels==cv2.GC_FGD)|(labels==cv2.GC_PR_FGD)
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    mask = components == 1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
    # Keep the visible handle hole; fill only pinholes introduced by segmentation.
    holes = binary_fill_holes(mask) & ~mask
    nc, hl, hs, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8))
    for j in range(1,nc):
        if hs[j,cv2.CC_STAT_AREA] < 30:
            mask[hl==j] = True
    if own_input:
        fixed = cv2.imread(str(committed/'mask.png'), cv2.IMREAD_GRAYSCALE)
        if fixed is None or fixed.shape != (h,w):
            raise ValueError('Committed mask must match the linear inputs')
        mask = fixed > 0
    cv2.imwrite(str(output/'composite.png'),display)
    save_image(output/'object_mask.png',mask.astype(float))
    overlay = display.copy()
    overlay[~mask] = overlay[~mask]//4
    cv2.drawContours(overlay,cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(0,255,0),1)
    cv2.imwrite(str(output/'segmentation_overlay.png'),overlay)
    # Intrinsics refer to full resized frame, then shift principal point for crop.
    x0,y0 = np.array(spec['roi'][:2])*width/1000
    metadata = {'raw_settings': meta, 'decode': 'dcraw -4 -h -r 1 1 1 1 -o 1',
                'full_resized_width':width, 'roi_preview1000':spec['roi'],
                'focal_pixels_assumed':55/23.5*width,
                'cx':width/2-x0, 'cy':(4000/6000*width)/2-y0,
                'intrinsics_note':'Approximate Nikon DX sensor width 23.5mm; centered principal point; not calibrated.',
                'segmentation':'Capture-specific silhouette seeds + GrabCut on maximum linear-light composite',
                'foreground_pixels':int(mask.sum())}
    (output/'capture_metadata.json').write_text(json.dumps(metadata,indent=2))
    return output, mask


def run(name,width):
    output, mask = prepare(name,width)
    images = load_images(output/'linear_inputs')
    B, residual = recover_integrable(images,2.,mask)
    foreground,inliers = make_masks(images,residual,output/'object_mask.png')
    samples = np.flatnonzero(inliers)
    if len(samples)>12000:
        samples=np.random.default_rng(0).choice(samples,12000,replace=False)
    best, score, history=entropy_search(B[:,samples],grid=7,levels=4)
    corrected=gbr_matrix(best).T@B
    initial=export_reconstruction(B,mask.shape,mask,output,'before',.1,masked=True)
    final=export_reconstruction(corrected,mask.shape,mask,output,'after',.1,masked=True)
    save_image(output/'fit_residual.png',np.where(mask,np.clip(residual/.5,0,1),0))
    save_image(output/'inlier_mask.png',inliers.astype(float))
    intensity=images[:,mask]
    singular=np.linalg.svd(intensity,compute_uv=False)
    report={'dataset':name,'shape':list(mask.shape),'foreground_pixels':int(mask.sum()),
            'inlier_pixels':int(inliers.sum()),'rank3_energy_fraction':float(np.sum(singular[:3]**2)/np.sum(singular**2)),
            'median_relative_rank3_residual':float(np.median(residual[mask])),
            'p90_relative_rank3_residual':float(np.quantile(residual[mask],.9)),
            'gbr_parameters':best.tolist(),'entropy':score,'search':history,
            'fraction_abs_nz_below_0.1':float(np.mean(abs(final[1][...,2][mask])<.1)),
            'finite_depth':bool(np.isfinite(final[2][mask]).all()),
            'limitations':['Uncalibrated reconstruction retains shape ambiguity; no ground-truth surface validation.',
                           'Gloss, shadows, interreflections, and spatially varying light violate the rank-3 Lambertian model.',
                           'Entropy assumes a favorable albedo distribution; cup logo and multicolored boot violate that prior.',
                           'Depth slopes clamped at abs(Nz)=0.1; depths have arbitrary scale and offset.']}
    if np.any(np.isclose(best,[-5,-5,.01])) or np.any(np.isclose(best,[5,5,5])):
        report['boundary_warning']='Entropy optimum hits a search bound; corrected shape is unreliable.'
    np.savez_compressed(output/'reconstruction.npz',B_before=B,B_after=corrected,mask=mask,inliers=inliers,
                         normals=final[1],albedo=final[0],depth=final[2],residual=residual)
    (output/'results.json').write_text(json.dumps(report,indent=2))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,8))
    panels=[(cv2.cvtColor(cv2.imread(str(output/'segmentation_overlay.png')),cv2.COLOR_BGR2RGB),'Segmentation'),
            ((final[1]+1)/2,'Normals after entropy GBR'),(np.where(mask,residual,np.nan),'Rank-3 relative residual'),
            (np.where(mask,initial[2],np.nan),'Depth before GBR'),(np.where(mask,final[2],np.nan),'Depth after GBR'),
            (cv2.cvtColor(cv2.imread(str(output/'after_surface.png')),cv2.COLOR_BGR2RGB),'Surface after GBR (arbitrary scale)')]
    for ax,(panel,title) in zip(axes.ravel(),panels):
        if panel.ndim==3 and panel.dtype.kind=='f': panel=np.where(mask[...,None],panel,0)
        ax.imshow(panel,cmap='viridis');ax.set_title(title);ax.axis('off')
    fig.suptitle(name+': uncalibrated photometric stereo; inspect limitations')
    fig.tight_layout();fig.savefig(output/'overview.png',dpi=130);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)
    meta = json.loads((output/'capture_metadata.json').read_text())
    subprocess.run([sys.executable,str(ROOT/'src'/'perspective.py'),
                    '--data',str(output/'linear_inputs'),'--output',str(output/'perspective'),
                    '--mask',str(output/'object_mask.png'),'--masked-domain',
                    '--focal',str(meta['focal_pixels_assumed']),
                    '--cx',str(meta['cx']),'--cy',str(meta['cy']),'--sigma','2'],check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',choices=list(SEEDS),default=list(SEEDS))
    parser.add_argument('--width',type=int,default=2000)
    args=parser.parse_args()
    for name in args.datasets: run(name,args.width)
