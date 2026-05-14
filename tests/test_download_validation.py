from pathlib import Path


def test_validate_download_rejects_html_named_as_archive(tmp_path: Path):
    from event_defect.downloads import validate_download_file

    fake = tmp_path / "metal_nut.tar.xz"
    fake.write_text("<!DOCTYPE html><title>not found</title>", encoding="utf-8")

    ok, reason = validate_download_file(fake, min_size_bytes=1024)

    assert ok is False
    assert "HTML" in reason


def test_count_image_files_counts_nested_pngs(tmp_path: Path):
    from event_defect.downloads import count_image_files

    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "x.png").write_bytes(b"png")
    (tmp_path / "a" / "b" / "x.txt").write_text("no", encoding="utf-8")

    assert count_image_files(tmp_path) == 1
