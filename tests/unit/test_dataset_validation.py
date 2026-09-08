import pytest

from dataset_validation import (
    DatasetValidationError,
    Example,
    check_split_leakage,
    load_jsonl,
    validate_examples,
)

KNOWN_LABELS = {"simple", "coding"}


class TestValidateExamples:
    def test_valid_dataset_has_no_errors(self):
        examples = [
            Example("what is 2+2", "simple"),
            Example("write a function", "coding"),
            Example("another simple prompt", "simple"),
            Example("another coding prompt", "coding"),
            Example("yet another simple prompt", "simple"),
        ]
        report = validate_examples(examples, KNOWN_LABELS, min_examples_per_label=1)
        assert report.is_valid

    def test_empty_text_is_error(self):
        examples = [Example("   ", "simple")]
        report = validate_examples(examples, KNOWN_LABELS)
        assert not report.is_valid
        assert any("empty" in e for e in report.errors)

    def test_missing_label_is_error(self):
        examples = [Example("some text", None)]
        report = validate_examples(examples, KNOWN_LABELS)
        assert not report.is_valid

    def test_unknown_label_is_error(self):
        examples = [Example("some text", "not_a_real_label")]
        report = validate_examples(examples, KNOWN_LABELS)
        assert not report.is_valid
        assert any("unknown label" in e for e in report.errors)

    def test_conflicting_duplicate_text_is_error(self):
        examples = [Example("same text", "simple"), Example("same text", "coding")]
        report = validate_examples(examples, KNOWN_LABELS)
        assert not report.is_valid
        assert any("conflicting labels" in e for e in report.errors)

    def test_missing_label_class_is_error(self):
        examples = [Example("only simple here", "simple")]
        report = validate_examples(examples, KNOWN_LABELS, min_examples_per_label=1)
        assert not report.is_valid
        assert any("'coding' has no training examples" in e for e in report.errors)

    def test_low_count_class_is_warning_not_error(self):
        examples = [Example(f"simple {i}", "simple") for i in range(5)] + [
            Example("only one coding example", "coding")
        ]
        report = validate_examples(examples, KNOWN_LABELS, min_examples_per_label=3)
        assert report.is_valid
        assert any("only 1 example" in w for w in report.warnings)

    def test_severe_imbalance_is_warning(self):
        examples = [Example(f"simple {i}", "simple") for i in range(20)] + [
            Example(f"coding {i}", "coding") for i in range(2)
        ]
        report = validate_examples(
            examples, KNOWN_LABELS, min_examples_per_label=1, max_imbalance_ratio=5.0
        )
        assert any("imbalance" in w for w in report.warnings)


class TestCheckSplitLeakage:
    def test_no_overlap_is_valid(self):
        train = [Example("train text", "simple")]
        test = [Example("test text", "simple")]
        report = check_split_leakage(train, test, names=["train", "test"])
        assert report.is_valid

    def test_overlap_is_error(self):
        train = [Example("shared text", "simple")]
        test = [Example("shared text", "simple")]
        report = check_split_leakage(train, test, names=["train", "test"])
        assert not report.is_valid
        assert any("leakage" in e for e in report.errors)


class TestLoadJsonl:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(DatasetValidationError):
            load_jsonl(tmp_path / "missing.jsonl")

    def test_invalid_json_line_raises(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"text": "ok", "label": "simple"}\nnot json\n', encoding="utf-8")
        with pytest.raises(DatasetValidationError):
            load_jsonl(path)

    def test_valid_file_loads(self, tmp_path):
        path = tmp_path / "good.jsonl"
        path.write_text(
            '{"text": "a", "label": "simple"}\n{"text": "b", "label": "coding"}\n',
            encoding="utf-8",
        )
        examples = load_jsonl(path)
        assert len(examples) == 2
        assert examples[0].text == "a"
        assert examples[0].label == "simple"

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "good.jsonl"
        path.write_text('{"text": "a", "label": "simple"}\n\n\n', encoding="utf-8")
        examples = load_jsonl(path)
        assert len(examples) == 1
