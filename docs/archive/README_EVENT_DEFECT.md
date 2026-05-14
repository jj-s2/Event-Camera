# Event-Camera Industrial Defect Detection

This workspace now includes a first runnable implementation of the event-camera
industrial defect detector proposed from the DBLP paper trail.

## What Was Built

- `event_defect/`: reusable package for event h5 loading, event tensor
  construction, mask-to-box conversion, model, and training losses.
- `scripts/list_dataset_sources.py`: prints the dataset sources I found and how
  each should be used.
- `scripts/build_event_manifest.py`: builds a manifest from real event-camera
  `.h5` or `.hdf5` files.
- `scripts/simulate_events_from_images.py`: converts industrial image anomaly
  datasets such as MVTec AD or NEU surface defects into synthetic conveyor-motion
  event streams.
- `scripts/train_event_defect.py`: trains the detector.
- `scripts/predict_event_defect.py`: runs one-sample inference and writes a
  heatmap.

## Algorithm

The implemented first version is `TimeAwareFusionDetector`:

1. Convert raw events into a polarity-split voxel grid plus a latest-time
   surface: `2 * bins + 1` channels.
2. Encode event features with a compact convolutional backbone.
3. Optionally fuse image-frame features with low-resolution time-aware
   cross-attention.
4. Predict a pixel-level defect heatmap, image-level defect class, and anomaly
   score.

This is intentionally deployment-friendly: it works with event-only industrial
h5 files first, while keeping the image branch ready for DAVIS-style cameras.

## Dataset Sources

Run:

```powershell
python scripts/list_dataset_sources.py
```

Primary event-native sources found:

- NSR BIVS Surface Defect Competition:
  https://academic.oup.com/nsr/article/10/6/nwad130/7158699
- JCDE aluminum substrate event-camera defect study:
  https://academic.oup.com/jcde/article/11/6/232/7861043

Important caveat: I found paper pages and detailed data descriptions, but not a
direct public download link for the BIVS/JCDE industrial event h5 files. The code
therefore supports the paper-described h5 shape directly, including `x`, `y`,
`t`, `p` and the JCDE-style `event_g`, `t`, `x`, `y` keys.

Practical bootstrap sources:

- MVTec AD:
  https://www.mvtec.com/company/research/datasets/mvtec-ad
- NEU Surface Defect Database:
  http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm

These are image datasets, not event-native datasets. Use the simulation script
to generate event streams by moving the image like a conveyor line.

## Real Event H5 Workflow

```powershell
python scripts/build_event_manifest.py `
  --root path\to\event_h5_root `
  --mask-root path\to\masks `
  --out data\event_defect_manifest.csv

python scripts/train_event_defect.py `
  --train-manifest data\event_defect_manifest.csv `
  --height 260 --width 346 --bins 5 `
  --epochs 20 --batch-size 8
```

## Image-to-Event Bootstrap Workflow

After downloading MVTec AD or NEU:

```powershell
python scripts/simulate_events_from_images.py `
  --root path\to\mvtec_or_neu `
  --out data\simulated_event_defects `
  --height 260 --width 346 `
  --steps 6 --shift-pixels 12 --threshold 0.12

python scripts/train_event_defect.py `
  --train-manifest data\simulated_event_defects\manifest.csv `
  --height 260 --width 346 --bins 5 `
  --epochs 20 --batch-size 8
```

## Prediction

```powershell
python scripts/predict_event_defect.py path\to\sample.h5 `
  --ckpt checkpoints_event_defect\best.pth `
  --heatmap-out outputs\sample_heatmap.png
```

## Notes

- The detector derives a multiclass map from the manifest while keeping `normal`
  fixed at class `0`.
- Defect localization uses masks when present. If no mask is available, defect
  samples can receive pseudo boxes from event-density connected components.
- The latest local improvement run is documented in
  `experiments/event_defect_improvement/RESULTS.md`.
- The next natural upgrade is contrastive defect codebook learning for
  `spot/scratch/stain`, matching the research design.
