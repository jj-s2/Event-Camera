# Event Defect Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable event-camera industrial defect detection baseline with dataset discovery, h5 ingestion, event representation, training, and prediction.

**Architecture:** Add a focused `event_defect` package beside the existing scripts. Keep original fall-detection files unchanged. Use real event h5 files as the primary format and image-to-event simulation as a bootstrap path for public industrial image datasets.

**Tech Stack:** Python, NumPy, h5py, OpenCV, PyTorch, pytest.

---

### Task 1: Core Event Data

**Files:**
- Create: `event_defect/data.py`
- Create: `event_defect/representation.py`
- Create: `event_defect/labels.py`
- Test: `tests/test_event_defect_core.py`

- [x] Write tests for h5 event loading, voxel grids, time surfaces, mask-to-box conversion, and dataset samples.
- [x] Run the tests and verify they fail because the package does not exist.
- [x] Implement the data loading and representation functions.
- [x] Run the tests and verify they pass.

### Task 2: Model

**Files:**
- Create: `event_defect/model.py`
- Test: `tests/test_event_defect_model.py`

- [x] Write tests for event-only and event-plus-image forward passes.
- [x] Run the tests and verify they fail because the model does not exist.
- [x] Implement `TimeAwareFusionDetector`.
- [x] Run the tests and verify they pass.

### Task 3: Training Utilities

**Files:**
- Create: `event_defect/training.py`
- Test: `tests/test_event_defect_training.py`

- [x] Write tests for box-to-heatmap conversion and combined loss computation.
- [x] Run the tests and verify they fail because training utilities do not exist.
- [x] Implement collate, heatmap target construction, loss computation, train, and eval helpers.
- [x] Run the tests and verify they pass.

### Task 4: Dataset and CLI Tools

**Files:**
- Create: `event_defect/dataset_sources.py`
- Create: `scripts/list_dataset_sources.py`
- Create: `scripts/build_event_manifest.py`
- Create: `scripts/simulate_events_from_images.py`
- Create: `scripts/train_event_defect.py`
- Create: `scripts/predict_event_defect.py`
- Test: `tests/test_dataset_sources.py`

- [x] Add dataset source metadata for event-native and surrogate industrial datasets.
- [x] Add manifest, simulation, training, and prediction scripts.
- [x] Verify dataset source tests pass.

### Task 5: Documentation and Verification

**Files:**
- Create: `README_EVENT_DEFECT.md`
- Create: `docs/event_defect_design.md`

- [x] Document found datasets, caveats, and commands.
- [x] Run the complete test suite.
- [x] Run Python compile validation.
- [x] Report verification evidence and remaining dataset access caveat.
