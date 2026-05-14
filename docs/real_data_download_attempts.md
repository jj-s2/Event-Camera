# Real Dataset Download Attempts

Date: 2026-05-13

## Target Dataset

Primary real dataset selected for this phase:

- `MSherbinii/mvtec-ad-metal-nut`
- Source: https://huggingface.co/datasets/MSherbinii/mvtec-ad-metal-nut
- Content: real MVTec AD metal nut subset with `train/good`, `test/good`,
  `test/bent`, `test/color`, `test/flip`, `test/scratch`, and ground-truth
  masks for defects.

Reason:

- Small enough to train locally compared with full MVTec AD.
- Real industrial defect imagery.
- Has defect masks for localization.
- Contains a `scratch` class relevant to the current failure mode.

## Successful Metadata Retrieval

The Hugging Face MCP returned dataset details, including the defect classes:

```text
bent, color, flip, scratch
```

A partial clone from `hf-mirror.com` succeeded far enough to retrieve:

```text
external/mvtec-ad-metal-nut-mirror/README.md
external/mvtec-ad-metal-nut-mirror/metal_nut/license.txt
external/mvtec-ad-metal-nut-mirror/metal_nut/readme.txt
```

`git lfs ls-files` also listed the real image/mask paths and LFS object IDs,
confirming the repository structure.

## Failed Local Image Downloads

### Hugging Face direct clone

Command:

```powershell
git clone --depth 1 https://huggingface.co/datasets/MSherbinii/mvtec-ad-metal-nut external\mvtec-ad-metal-nut
```

Result:

```text
Failed to connect to huggingface.co port 443
```

### Hugging Face parquet direct download

Command:

```powershell
curl https://huggingface.co/datasets/MSherbinii/mvtec-ad-metal-nut/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet
```

Result:

```text
Failed to connect to huggingface.co port 443
```

### hf-mirror parquet download

Command:

```powershell
curl https://hf-mirror.com/datasets/MSherbinii/mvtec-ad-metal-nut/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet
```

Result:

```text
schannel: AcquireCredentialsHandle failed: SEC_E_NO_CREDENTIALS
```

### MVTec official mydrive direct URL

Command:

```powershell
curl https://www.mydrive.ch/.../metal_nut.tar.xz
```

Result:

```text
Downloaded 10857 bytes of HTML, not an archive.
Validation: external\downloads\metal_nut.tar.xz is HTML, not a dataset archive.
```

### Downloader script

Command:

```powershell
python scripts/download_mvtec_metal_nut.py --target external\mvtec-ad-metal-nut-real
```

Result:

```text
fatal: unable to access 'https://hf-mirror.com/datasets/MSherbinii/mvtec-ad-metal-nut/':
schannel: AcquireCredentialsHandle failed: SEC_E_NO_CREDENTIALS
```

## Successful Download After Proxy Configuration

The local Clash proxy exposed:

```text
http://127.0.0.1:7897
```

Using explicit Git proxy and OpenSSL backend, the real Hugging Face LFS content
was downloaded into:

```text
external/mvtec-ad-metal-nut-real
```

Commands used:

```powershell
$env:GIT_LFS_SKIP_SMUDGE='1'
git -c http.proxy=http://127.0.0.1:7897 -c https.proxy=http://127.0.0.1:7897 -c http.sslBackend=openssl clone --depth 1 https://huggingface.co/datasets/MSherbinii/mvtec-ad-metal-nut external\mvtec-ad-metal-nut-real
git -C external\mvtec-ad-metal-nut-real -c http.proxy=http://127.0.0.1:7897 -c https.proxy=http://127.0.0.1:7897 -c http.sslBackend=openssl lfs pull -I "metal_nut/test/good/**,metal_nut/test/scratch/**,metal_nut/ground_truth/scratch/**,metal_nut/train/good/**"
git -C external\mvtec-ad-metal-nut-real -c http.proxy=http://127.0.0.1:7897 -c https.proxy=http://127.0.0.1:7897 -c http.sslBackend=openssl lfs pull -I "metal_nut/test/bent/**,metal_nut/test/color/**,metal_nut/test/flip/**,metal_nut/ground_truth/bent/**,metal_nut/ground_truth/color/**,metal_nut/ground_truth/flip/**"
```

Validated local PNG counts:

```text
train/good: 220
test/good: 22
test/bent: 25
test/color: 22
test/flip: 23
test/scratch: 23
ground_truth/bent: 25
ground_truth/color: 22
ground_truth/flip: 23
ground_truth/scratch: 23
```

The real images are regular PNG files of roughly 444 KB to 516 KB each, not LFS
pointer placeholders.

## Reproduction Commands

The real image data was converted to DVS-style event streams from a controlled
horizontal scan. This is real industrial defect imagery, but the event streams
are simulated from images rather than captured by a physical event camera:

```powershell
python scripts\simulate_events_from_images.py --root external\mvtec-ad-metal-nut-real --out experiments\mvtec_metal_nut_real_events --height 96 --width 96 --steps 7 --shift-pixels 9 --threshold 0.075
python scripts\split_event_manifest.py --manifest experiments\mvtec_metal_nut_real_events\manifest.csv --out-dir experiments\mvtec_metal_nut_real_events\splits --val-ratio 0.2 --test-ratio 0.2 --seed 42
python scripts\binarize_event_manifest.py --manifest experiments\mvtec_metal_nut_real_events\splits\train.csv --out experiments\mvtec_metal_nut_real_events\splits_binary\train.csv
python scripts\binarize_event_manifest.py --manifest experiments\mvtec_metal_nut_real_events\splits\val.csv --out experiments\mvtec_metal_nut_real_events\splits_binary\val.csv
python scripts\binarize_event_manifest.py --manifest experiments\mvtec_metal_nut_real_events\splits\test.csv --out experiments\mvtec_metal_nut_real_events\splits_binary\test.csv
```
