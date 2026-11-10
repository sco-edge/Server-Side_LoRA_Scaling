import pytest


@pytest.mark.parametrize("row,reason", [
    ("{", "Expecting"), ("[]", "JSON object"), ('{"label": 0}', "text"),
    ('{"text":"x","label":false}', "integer class label"),
    ('{"text":"x","label":2}', "0..1"),
])
def test_record_diagnostics(sls, tmp_path, row, reason):
    path = tmp_path / "data.jsonl"
    path.write_text('{"text":"ok","label":0}\n\n' + row, encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        sls.load_data(path, "{text}", 2)
    assert f"{path}:3:" in str(error.value)
    assert reason in str(error.value)
