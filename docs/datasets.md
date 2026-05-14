# Datasets

## Native Event-Camera Targets

The code supports event-native H5 files with common event-camera layouts, but
direct public downloads for the industrial event-native papers were not exposed
during this run.

Relevant targets and references:

- NSR BIVS Surface Defect Competition
- JCDE aluminum substrate event-camera defect study
- Prophesee GEN1 for non-industrial event encoder sanity checks

Run:

```powershell
python scripts\list_dataset_sources.py
```

## Real Industrial Dataset Used Locally

Primary local dataset:

- Source: https://hf.co/datasets/MSherbinii/mvtec-ad-metal-nut
- Local directory: `external/mvtec-ad-metal-nut-real`
- Classes: `good`, `bent`, `color`, `flip`, `scratch`
- Masks: available for defect classes

This dataset is image-based. Event streams in this repository are simulated from
the real images using `scripts/simulate_events_from_images.py`.

## Reproduction

Convert images to DVS-style event streams:

```powershell
python scripts\simulate_events_from_images.py `
  --root external\mvtec-ad-metal-nut-real `
  --out experiments\mvtec_metal_nut_real_events `
  --height 96 --width 96 `
  --steps 7 --shift-pixels 9 --threshold 0.075
```

Create splits:

```powershell
python scripts\split_event_manifest.py `
  --manifest experiments\mvtec_metal_nut_real_events\manifest.csv `
  --out-dir experiments\mvtec_metal_nut_real_events\splits `
  --val-ratio 0.2 --test-ratio 0.2 --seed 42
```

Binarize labels:

```powershell
python scripts\binarize_event_manifest.py --manifest experiments\mvtec_metal_nut_real_events\splits\train.csv --out experiments\mvtec_metal_nut_real_events\splits_binary\train.csv
python scripts\binarize_event_manifest.py --manifest experiments\mvtec_metal_nut_real_events\splits\val.csv --out experiments\mvtec_metal_nut_real_events\splits_binary\val.csv
python scripts\binarize_event_manifest.py --manifest experiments\mvtec_metal_nut_real_events\splits\test.csv --out experiments\mvtec_metal_nut_real_events\splits_binary\test.csv
```

Detailed download history is kept in
[real_data_download_attempts.md](real_data_download_attempts.md).
