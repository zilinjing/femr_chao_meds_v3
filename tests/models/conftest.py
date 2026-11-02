"""Shared fixtures: a toy meds_reader database built with meds_reader_convert."""

import datetime
import json
import subprocess

import meds
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _toy_rows():
    rows = []
    for subject_id in range(20):
        birth = datetime.datetime(1980, 1, 1)
        rows.append((subject_id, birth, meds.birth_code, None))
        visit = datetime.datetime(2020, 1, 1)
        if subject_id < 6:
            rows.append((subject_id, visit, "C", None))
        elif subject_id < 12:
            rows.append((subject_id, visit, "D", None))
        if subject_id < 15:
            # > 10,000 lab values in total, so the reservoir sampler has to make random choices.
            for i in range(800):
                rows.append((subject_id, visit + datetime.timedelta(hours=i + 1), "L", float((subject_id * 800 + i) % 997)))
    return rows


@pytest.fixture(scope="module")
def toy_database(tmp_path_factory):
    root = tmp_path_factory.mktemp("toy")
    source = root / "meds"
    (source / "data").mkdir(parents=True)
    (source / "metadata").mkdir()

    subject_id, time, code, value = zip(*_toy_rows())
    table = pa.table(
        {"subject_id": subject_id, "time": time, "code": code, "numeric_value": value},
        schema=pa.schema(
            [("subject_id", pa.int64()), ("time", pa.timestamp("us")), ("code", pa.string()), ("numeric_value", pa.float32())]
        ),
    )
    pq.write_table(table, source / "data" / "0.parquet")
    pq.write_table(
        pa.table({"code": pa.array([], pa.string()), "description": pa.array([], pa.string()),
                  "parent_codes": pa.array([], pa.list_(pa.string()))}),
        source / "metadata" / "codes.parquet",
    )
    with open(source / "metadata" / "dataset.json", "w") as f:
        json.dump({"dataset_name": "toy", "dataset_version": "1", "etl_name": "test", "etl_version": "1",
                   "meds_version": meds.__version__}, f)

    database = root / "reader"
    subprocess.run(["meds_reader_convert", str(source), str(database)], check=True)
    return str(database)
