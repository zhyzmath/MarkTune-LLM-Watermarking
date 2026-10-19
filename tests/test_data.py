import pytest

from marktune.data import build_evaluation_rows, split_documents


class WordTokenizer:
    def encode(self, text, **kwargs):
        return text.split()

    def decode(self, tokens, **kwargs):
        return " ".join(tokens)


def test_document_splits_are_disjoint_deduplicated_and_seeded():
    rows = [{"text": f"document {i} " + "word " * 12} for i in range(20)]
    rows += [rows[0], rows[0]]
    counts = {"train": 8, "validation": 3, "attack-train": 4}
    first = split_documents(rows, WordTokenizer(), counts, 42, 10)
    second = split_documents(rows, WordTokenizer(), counts, 42, 10)
    assert first == second
    ids = [row["document_id"] for split in first.values() for row in split]
    assert len(ids) == len(set(ids)) == 15
    with pytest.raises(ValueError, match="unique eligible"):
        split_documents(rows[:2], WordTokenizer(), counts, 42, 10)


def test_c4_prompt_and_reference_do_not_overlap():
    rows = [{"text": " ".join(str(i) for i in range(401))}]
    item = build_evaluation_rows(rows, WordTokenizer(), "c4", 1)[0]
    assert item["prompt"].split() == [str(i) for i in range(200)]
    assert item["reference"].split() == [str(i) for i in range(200, 400)]
    with pytest.raises(ValueError):
        build_evaluation_rows([{"text": " ".join(str(i) for i in range(400))}], WordTokenizer(), "c4", 1)


def test_eli5_keeps_question_and_truncates_answer():
    rows = [{"question": "Why?", "answer": "word " * 201}]
    item = build_evaluation_rows(rows, WordTokenizer(), "eli5", 1)[0]
    assert item["prompt"] == "Why?"
    assert len(item["reference"].split()) == 200
