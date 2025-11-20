import gzip
import json
from pathlib import Path

from plume_nav_sim.data_capture.recorder import RunRecorder


def _write_steps_file(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec))
            fh.write("\n")


def test_seed_summary_no_files(tmp_path: Path) -> None:
    rec = RunRecorder(tmp_path, experiment="seed-summary-no-files")
    assert rec._seed_summary() == {}


def test_seed_summary_single_file(tmp_path: Path) -> None:
    rec = RunRecorder(tmp_path, experiment="seed-summary-single")
    steps_path = rec.root / "steps.jsonl.gz"
    _write_steps_file(
        steps_path,
        [
            {"step": 0, "seed": 111},
            {"step": 1},
            {"step": 2, "seed": 222},
            {"step": 3, "seed": 111},
        ],
    )

    summary = rec._seed_summary()
    assert summary == {"unique_count": 2, "min": 111, "max": 222}


def test_seed_summary_multiple_shards(tmp_path: Path) -> None:
    rec = RunRecorder(tmp_path, experiment="seed-summary-multi")
    base = rec.root / "steps.jsonl.gz"
    part1 = rec.root / "steps.part1.jsonl.gz"
    part2 = rec.root / "steps.part2.jsonl.gz"

    _write_steps_file(base, [{"seed": 10}, {"seed": 30}])
    _write_steps_file(part1, [{"seed": 20}])
    _write_steps_file(part2, [{"seed": 40}, {}])

    summary = rec._seed_summary()
    assert summary == {"unique_count": 4, "min": 10, "max": 40}


def test_seed_summary_ignores_corrupt_lines(tmp_path: Path) -> None:
    rec = RunRecorder(tmp_path, experiment="seed-summary-corrupt-lines")
    steps_path = rec.root / "steps.jsonl.gz"
    steps_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(steps_path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"seed": 5}))
        fh.write("\n")
        fh.write("this is not json\n")
        fh.write(json.dumps({"seed": 7}))
        fh.write("\n")
        fh.write(json.dumps({"seed": "not-int"}))
        fh.write("\n")

    summary = rec._seed_summary()
    assert summary == {"unique_count": 2, "min": 5, "max": 7}


def test_seed_summary_skips_unreadable_files(tmp_path: Path) -> None:
    rec = RunRecorder(tmp_path, experiment="seed-summary-unreadable-file")
    bad = rec.root / "steps.jsonl.gz"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("not gzipped content", encoding="utf-8")

    good = rec.root / "steps.part1.jsonl.gz"
    _write_steps_file(good, [{"seed": 123}])

    summary = rec._seed_summary()
    assert summary == {"unique_count": 1, "min": 123, "max": 123}
