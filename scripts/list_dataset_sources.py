from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from event_defect.dataset_sources import recommended_sources


def main() -> None:
    for idx, source in enumerate(recommended_sources(), start=1):
        native = "event-native" if source.event_native else "image-to-event surrogate"
        industrial = "industrial" if source.industrial else "non-industrial"
        print(f"{idx}. {source.name}")
        print(f"   URL: {source.url}")
        print(f"   Type: {native}, {industrial}")
        print(f"   Access: {source.access}")
        print(f"   Role: {source.role}")
        print(f"   Notes: {source.notes}")
        print()


if __name__ == "__main__":
    main()
