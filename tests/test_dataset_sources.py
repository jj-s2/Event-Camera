def test_dataset_registry_contains_real_event_and_surrogate_sources():
    from event_defect.dataset_sources import DATASET_SOURCES, recommended_sources

    names = {source.name for source in DATASET_SOURCES}

    assert "NSR BIVS Surface Defect Competition" in names
    assert "MVTec AD Event Simulation Surrogate" in names
    assert any(source.event_native for source in DATASET_SOURCES)
    assert recommended_sources()[0].event_native is True
