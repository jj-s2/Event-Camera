from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSource:
    name: str
    url: str
    event_native: bool
    industrial: bool
    access: str
    role: str
    notes: str


DATASET_SOURCES = [
    DatasetSource(
        name="NSR BIVS Surface Defect Competition",
        url="https://academic.oup.com/nsr/article/10/6/nwad130/7158699",
        event_native=True,
        industrial=True,
        access="paper-described; direct public download was not exposed in the paper page",
        role="primary target format",
        notes="Bio-inspired vision sensor data for spot, scratch and stain defects on aluminum substrates.",
    ),
    DatasetSource(
        name="JCDE Aluminum Substrate Event Defect Study",
        url="https://academic.oup.com/jcde/article/11/6/232/7861043",
        event_native=True,
        industrial=True,
        access="paper-described; use as method and format reference",
        role="algorithm reference",
        notes="Uses event aggregation, pseudo labels and a correction network for aluminum substrate defects.",
    ),
    DatasetSource(
        name="MVTec AD Event Simulation Surrogate",
        url="https://www.mvtec.com/company/research/datasets/mvtec-ad",
        event_native=False,
        industrial=True,
        access="public download after accepting dataset terms",
        role="bootstrap training and smoke tests",
        notes="Industrial anomaly images and masks can be converted to synthetic conveyor-motion events.",
    ),
    DatasetSource(
        name="NEU Surface Defect Database Event Simulation Surrogate",
        url="http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm",
        event_native=False,
        industrial=True,
        access="public academic dataset page",
        role="class-balanced surface-defect pretraining",
        notes="Steel surface classes can be converted to synthetic event streams for pretraining.",
    ),
    DatasetSource(
        name="Prophesee GEN1 Automotive Detection",
        url="https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/",
        event_native=True,
        industrial=False,
        access="public event-camera benchmark",
        role="event encoder pretraining only",
        notes="Not industrial defects, but useful for event representation sanity checks.",
    ),
]


def recommended_sources() -> list[DatasetSource]:
    """Return sources in the order they should be attempted."""
    return sorted(
        DATASET_SOURCES,
        key=lambda item: (
            not item.event_native,
            not item.industrial,
            item.name,
        ),
    )
