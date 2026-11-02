"""Tests for the shared Tokenizer (femr.models.tokenizer).

The toy dataset has an ontology parent P with children C and D. Only C and D occur as raw codes,
so P is an input token (ancestor) but must not be a next-code target in original_codes.
"""

import datetime
import json
import pathlib
import subprocess

import meds
import meds_reader
import msgpack
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from femr.models.tokenizer import CLMBRTokenizer, HierarchicalTokenizer, Tokenizer

PARENTS = {"C": {"P"}, "D": {"P"}}


class ToyOntology:
    def get_parents(self, code):
        return PARENTS.get(code, set())

    def get_all_parents(self, code):
        result = {code}
        for parent in self.get_parents(code):
            result |= self.get_all_parents(parent)
        return result


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


def _train(database, num_threads=1, vocab_size=100_000):
    with meds_reader.SubjectDatabase(database, num_threads=num_threads) as db:
        return Tokenizer.train(db, vocab_size=vocab_size, ontology=ToyOntology())


def _vocab_codes(tokenizer):
    return [entry["code_string"] for entry in tokenizer.dictionary["vocab"] if entry["type"] == "code"]


def test_ancestor_is_an_input_token_but_not_a_target(toy_database):
    tokenizer = _train(toy_database)

    assert {"P", "C", "D", "L"} <= set(_vocab_codes(tokenizer))
    original_codes = tokenizer.dictionary["original_codes"]
    assert "P" not in original_codes
    assert {"C", "D", "L"} <= set(original_codes)
    # Targets follow vocabulary order.
    assert original_codes == [code for code in _vocab_codes(tokenizer) if code in set(original_codes)]


def test_original_codes_are_computed_after_the_vocab_size_cut(toy_database):
    full = _train(toy_database)
    vocab_size = len(full.dictionary["vocab"]) - 1
    cut = _train(toy_database, vocab_size=vocab_size)

    assert len(cut.dictionary["vocab"]) == vocab_size
    assert set(cut.dictionary["original_codes"]) <= set(_vocab_codes(cut))


def test_training_is_reproducible_across_runs(toy_database):
    first = _train(toy_database, num_threads=2)
    second = _train(toy_database, num_threads=2)

    assert msgpack.dumps(first.dictionary) == msgpack.dumps(second.dictionary)


def test_old_dictionary_without_original_codes_still_loads(toy_database, tmp_path):
    dictionary = dict(_train(toy_database).dictionary)
    del dictionary["original_codes"]
    with open(tmp_path / "dictionary.msgpack", "wb") as f:
        msgpack.dump(dictionary, f)

    tokenizer = Tokenizer.from_pretrained(tmp_path, ontology=ToyOntology())
    assert tokenizer.vocab_size == len(dictionary["vocab"])
    with pytest.raises(RuntimeError):
        tokenizer.get_event_code_token(type("Event", (), {"code": "C"})())


def test_banned_properties_get_no_tokens(toy_database):
    with meds_reader.SubjectDatabase(toy_database, num_threads=1) as db:
        tokenizer = Tokenizer.train(db, vocab_size=100_000, ontology=ToyOntology(), banned_properties={"numeric_value"})
        assert "numeric_value" in db.properties  # the database itself is not modified

    assert not [e for e in tokenizer.dictionary["vocab"] if e.get("property") == "numeric_value"]
    assert [e for e in _train(toy_database).dictionary["vocab"] if e.get("property") == "numeric_value"]


def test_old_names_are_aliases():
    assert HierarchicalTokenizer is Tokenizer
    assert CLMBRTokenizer is Tokenizer
