import numpy as np


def test_patch_score_topk_and_quantile():
    from scripts.train_patchcore_event_defect import patch_score

    distances = np.array([0.1, 0.5, 0.2, 0.9, 0.4])

    assert patch_score(distances, "max") == 0.9
    assert patch_score(distances, "top2") == 0.7
    assert patch_score(distances, "q50") == 0.4


def test_select_threshold_prefers_high_accuracy_and_f1():
    from scripts.train_patchcore_event_defect import select_threshold, summarize_predictions

    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    threshold, metrics = select_threshold(y_true, y_score)
    pred = (y_score >= threshold).astype(np.int64)

    assert summarize_predictions(y_true, pred) == metrics
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
