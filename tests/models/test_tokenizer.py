"""Tests for the shared Tokenizer (femr.models.tokenizer).

The toy dataset has an ontology parent P with children C and D. Only C and D occur as raw codes,
so P is an input token (ancestor) but must not be a next-code target in original_codes.
"""

import meds_reader
import msgpack
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
