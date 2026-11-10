import pytest


@pytest.mark.parametrize("length", ["0", "-1"])
def test_invalid_length_fails_before_reading_data(sls, length, capsys):
    with pytest.raises(SystemExit) as error:
        sls.main(["--model", "m", "--adapter", "a", "--data", "missing.jsonl",
                  "--task", "sst2", "--max-len", length])
    assert error.value.code == 2
    assert "--max-len" in capsys.readouterr().err
