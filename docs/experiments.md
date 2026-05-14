# Experiments

## MVTec Metal Nut Real-Data Run

Report:
`experiments/mvtec_metal_nut_real_events/REPORT.md`

The dataset contains real industrial defect images. Event streams are simulated
from the images through controlled scan motion.

## Split

- train: 203 samples
- validation: 65 samples
- test: 67 samples

Binary test split:

- normal: 48
- defect: 19

## Results

| Method | Input | Test Accuracy | Precision | Recall | F1 | Confusion Matrix |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Event CNN | simulated DVS | 83.58% | 66.67% | 84.21% | 74.42% | TN=40, FP=8, FN=3, TP=16 |
| Feature ensemble | simulated DVS | 86.57% | 81.25% | 68.42% | 74.29% | TN=45, FP=3, FN=6, TP=13 |
| Feature ensemble | simulated DVS + APS | 95.52% | 90.00% | 94.74% | 92.31% | TN=46, FP=2, FN=1, TP=18 |
| PatchCore normal memory | APS | 100.00% | 100.00% | 100.00% | 100.00% | TN=48, FP=0, FN=0, TP=19 |

## Best Command

```powershell
python scripts\train_patchcore_event_defect.py `
  --train-manifest experiments\mvtec_metal_nut_real_events\splits_binary\train.csv `
  --val-manifest experiments\mvtec_metal_nut_real_events\splits_binary\val.csv `
  --test-manifest experiments\mvtec_metal_nut_real_events\splits_binary\test.csv `
  --input-mode aps `
  --height 96 --width 96 --bins 4 `
  --memory-sizes 5000,10000,20000,40000,80000 `
  --score-names max,top5,top10,q99,q95 `
  --out-dir experiments\mvtec_metal_nut_real_events\patchcore_aps
```

## Caveat

The 98% target is reached by the APS branch. Pure event-only branches remain
below 98% on this split. Use this distinction when describing the result.
