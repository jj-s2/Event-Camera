# Event Defect Detector Design

## Goal

Build an event-camera industrial defect detector that can train on real
event-native h5 streams when available, and can bootstrap from public industrial
image anomaly datasets by simulating conveyor-motion event streams.

## Research Basis

- DBLP author page used as the seed:
  https://dblp.org/pid/348/4679.html
- Event-guided rolling shutter correction contributes time-aware cross-modal
  attention.
- Neural image re-exposure contributes the idea of treating events as a
  controllable time-window visual representation.
- Multi-stage event multimodal tracking contributes staged fusion.
- CCL-LGS contributes the future codebook/contrastive prototype direction.
- NSR and JCDE event-camera surface defect papers provide the industrial defect
  data format and deployment target.

## First Runnable Version

The first implementation is intentionally narrow:

- Input: event h5 files with `x/y/t/p`, `events`, or JCDE-style
  `event_g/t/x/y` keys.
- Representation: polarity-split voxel grid plus latest-time surface.
- Model: compact event CNN with optional time-aware image fusion branch.
- Outputs: heatmap, binary class logits, anomaly score.
- Training: supervised heatmap/class/anomaly loss.
- Dataset fallback: synthetic events from MVTec AD or NEU images.

## Deferred Upgrades

- Multiclass defect labels for spot, scratch, stain, pit and crack.
- Contrastive codebook loss for defect prototypes.
- Semi-supervised pseudo-label mining.
- Motion compensation using conveyor encoder speed or event optical flow.
- ONNX export and video-line online inference.
