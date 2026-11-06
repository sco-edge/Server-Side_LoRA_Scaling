import json

import pytest


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def test_formats_template_and_strips_fields(sls, tmp_path):
    data = _write(tmp_path / "d.jsonl", [{"sentence": "  Great movie.  ", "label": 1}])
    prompts, y = sls.load_data(data, "{sentence} It was", 2)
    assert prompts == ["Great movie. It was"]
    assert y.tolist() == [1]


def test_rejects_out_of_range_labels(sls, tmp_path):
    data = _write(tmp_path / "d.jsonl", [{"sentence": "x", "label": 2}])
    with pytest.raises(SystemExit):
        sls.load_data(data, "{sentence}", 2)


def test_rejects_empty_file(sls, tmp_path):
    data = tmp_path / "d.jsonl"
    data.write_text("\n\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        sls.load_data(str(data), "{sentence}", 2)


def test_keeps_non_string_fields(sls, tmp_path):
    data = _write(tmp_path / "d.jsonl", [{"text": " a ", "n": 3, "label": 0}])
    prompts, _ = sls.load_data(data, "{text} n={n}", 2)
    assert prompts == ["a n=3"]
