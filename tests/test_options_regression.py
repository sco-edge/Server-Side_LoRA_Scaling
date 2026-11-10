import pytest


@pytest.mark.parametrize("custom", [
    ["--template", "{text}"],
    ["--label-words", "good", "bad"],
    ["--template", "{text}", "--label-words", "good", "bad"],
])
def test_task_conflicts_with_custom_options(sls, custom, capsys):
    with pytest.raises(SystemExit) as error:
        sls.main(["--model", "m", "--adapter", "a", "--data", "d", "--task", "sst2", *custom])
    assert error.value.code == 2
    assert "--task cannot be combined" in capsys.readouterr().err
