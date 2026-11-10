import json

import pytest


@pytest.mark.parametrize("value", [1.9, True, False, "1.9"])
def test_rejects_non_integer_labels(sls, tmp_path, value):
    path = tmp_path / "sample.jsonl"
    path.write_text(json.dumps({"text": "hello", "label": value}), encoding="utf-8")
    with pytest.raises(SystemExit, match="integer class label"):
        sls.load_data(path, "{text}", 2)


@pytest.mark.parametrize("value", [1, "1"])
def test_accepts_integer_labels(sls, tmp_path, value):
    path = tmp_path / "sample.jsonl"
    path.write_text(json.dumps({"text": "hello", "label": value}), encoding="utf-8")
    _, labels = sls.load_data(path, "{text}", 2)
    assert labels.tolist() == [1]
