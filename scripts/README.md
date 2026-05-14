# Scripts

Command-line workflows for the event defect project.

## Data Preparation

- `list_dataset_sources.py`: print known dataset sources.
- `download_mvtec_metal_nut.py`: helper for downloading the MVTec metal nut
  subset.
- `simulate_events_from_images.py`: convert image datasets to simulated
  DVS-style event H5 files.
- `build_event_manifest.py`: build a manifest from existing event H5 files.
- `split_event_manifest.py`: create stratified train/validation/test splits.
- `binarize_event_manifest.py`: map multiclass labels to `normal`/`defect`.

## Training and Evaluation

- `train_event_defect.py`: train the event CNN baseline.
- `evaluate_event_defect.py`: evaluate an event CNN checkpoint.
- `train_event_defect_feature_ensemble.py`: train feature baselines.
- `train_patchcore_event_defect.py`: train the PatchCore-style normal-memory
  detector.

## Prediction and Demos

- `predict_event_defect.py`: run event CNN inference on one H5 sample.
- `demo_patchcore_detection.py`: generate a visual PatchCore detection panel for
  one manifest sample.
