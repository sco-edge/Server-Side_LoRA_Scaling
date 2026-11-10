import pytest


@pytest.mark.parametrize(("row", "reason"), [
    ("{", "Expecting"),
    ('{"label": 0}', "sentence"),
    ('{"sentence": "ok"}', "label"),
    ("[]", "JSON object"),
])
def test_bad_row_reports_source_location(sls, tmp_path, row, reason):
    path = tmp_path / "data.jsonl"
    path.write_text('{"sentence":"ok","label":0}\n\n' + row, encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        sls.load_data(path, "{sentence}", 2)
    assert f"{path}:3:" in str(error.value)
    assert reason in str(error.value)
