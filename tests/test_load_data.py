import json


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def test_formats_template_and_strips_fields(sls, tmp_path):
    data = _write(tmp_path / "d.jsonl", [{"sentence": "  Great movie.  ", "label": 1}])
    prompts, y = sls.load_data(data, "{sentence} It was", 2)
    assert prompts == ["Great movie. It was"]
    assert y.tolist() == [1]
