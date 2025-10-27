import gzip
from pathlib import Path

from plume_nav_sim.data_capture.writer import JSONLGzWriter


def test_writer_writes_gzipped_lines(tmp_path: Path):
    path = tmp_path / "steps"
    w = JSONLGzWriter(path)
    w.write_obj({"a": 1})
    w.write_obj({"b": 2})
    w.flush()
    w.close()

    gz = (
        path.with_suffix(path.suffix + ".jsonl.gz")
        if path.suffix
        else Path(str(path) + ".jsonl.gz")
    )
    assert gz.exists()
    with gzip.open(gz, "rt", encoding="utf-8") as fh:
        lines = [line.strip() for line in fh if line.strip()]
    assert len(lines) == 2


def test_writer_rotates_by_size(tmp_path: Path):
    path = tmp_path / "episodes"
    w = JSONLGzWriter(path, rotate_size_bytes=200)
    for i in range(200):
        w.write_obj({"i": i, "payload": "x" * 20})
    w.close()
    # base + parts expected
    found = sorted(tmp_path.glob("episodes*.jsonl.gz"))
    assert len(found) >= 1
