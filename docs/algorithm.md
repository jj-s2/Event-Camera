# Algorithm Notes

## Problem

Detect industrial defects from event-camera data, with a practical path for
DAVIS-style cameras that provide both event streams and APS/intensity frames.

## Event Representation

Raw events are loaded from H5 files and converted into:

- polarity-split voxel grids,
- latest-time surfaces,
- optional event RGB visualizations for diagnostics.

Supported H5 layouts include:

- `events` arrays shaped like `[x, y, t, p]`,
- separate `x`, `y`, `t`, `p` datasets,
- JCDE-style `event_g`, `t`, `x`, `y` datasets.

## Event CNN Baseline

`TimeAwareFusionDetector` is a compact CNN with:

- event encoder,
- optional image encoder and cross-attention fusion,
- heatmap head,
- class head,
- anomaly score head.

It is useful for event-only research and localization experiments, but on the
current MVTec metal nut simulation split it does not reach 98% accuracy.

## Feature Ensemble Baseline

The feature ensemble extracts:

- event density and temporal statistics,
- APS/image texture statistics,
- edge and color histograms.

It gives a stronger classical baseline and reached 95.52% on the local real-data
test split when using event + APS features.

## PatchCore Normal-Memory Detector

The strongest current branch is `train_patchcore_event_defect.py`.

It builds a memory bank of pretrained ResNet-18 patch descriptors from normal
training samples only. At inference time, the anomaly score is the mean of the
largest nearest-normal-patch distances. The threshold is selected using the
validation split only.

Current selected configuration:

- input mode: `aps`
- memory size: `10000`
- score: `top5`
- test accuracy: `100.00%` on the held-out local test split

This branch is best interpreted as an APS-assisted industrial inspection method
for event-camera rigs that expose intensity frames.
