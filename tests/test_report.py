import csv
import json

import numpy as np

from septosympto.leaf import Leaf
from septosympto.measure import LeafMeasurement, measure_leaf
from septosympto.report import build_manifest, file_sha256, write_csv, write_manifest


def a_measurement(index=1) -> LeafMeasurement:
    mask = np.ones((100, 1000), bool)
    leaf = Leaf(image="scan", leaf_index=index, bbox=(0, 0, 1000, 100), mask=mask, qc=())
    return measure_leaf(leaf, np.zeros((100, 1000), bool), None, 472.44)


def test_csv_header_matches_the_dataclass(tmp_path):
    path = tmp_path / "results.csv"
    write_csv([a_measurement()], path)
    with path.open() as handle:
        header = next(csv.reader(handle))
    assert header == LeafMeasurement.columns()


def test_every_row_has_exactly_one_value_per_column(tmp_path):
    """The v1 --import bug wrote 23 headers and 19 values; this makes that impossible."""
    path = tmp_path / "results.csv"
    write_csv([a_measurement(1), a_measurement(2)], path)
    with path.open() as handle:
        rows = list(csv.reader(handle))
    n_cols = len(rows[0])
    assert all(len(row) == n_cols for row in rows[1:])
    assert len(rows) == 3


def test_write_csv_returns_the_row_count(tmp_path):
    path = tmp_path / "results.csv"
    assert write_csv([a_measurement(1), a_measurement(2)], path) == 2


def test_manifest_records_weights_hash_and_provenance(tmp_path):
    weights = tmp_path / "necrosis.safetensors"
    weights.write_bytes(b"not real weights")

    manifest = build_manifest(
        weights={"necrosis": weights},
        parameters={"necrosis_threshold": 0.8},
        n_scans=7,
        n_leaves=27,
        timestamp="2026-07-10T00:00:00+00:00",
    )
    assert manifest["n_scans"] == 7
    assert manifest["n_leaves"] == 27
    assert manifest["parameters"]["necrosis_threshold"] == 0.8
    assert manifest["weights"]["necrosis"]["sha256"] == file_sha256(weights)
    assert "septosympto_version" in manifest


def test_manifest_round_trips_through_json(tmp_path):
    weights = tmp_path / "w.safetensors"
    weights.write_bytes(b"x")
    manifest = build_manifest(
        weights={"necrosis": weights}, parameters={}, n_scans=1, n_leaves=1,
        timestamp="2026-07-10T00:00:00+00:00",
    )
    path = tmp_path / "run.manifest.json"
    write_manifest(manifest, path)
    assert json.loads(path.read_text())["timestamp"] == "2026-07-10T00:00:00+00:00"
