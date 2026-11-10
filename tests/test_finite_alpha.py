import pytest


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_alpha_fails_before_loading_data(sls, capsys, value):
    with pytest.raises(SystemExit) as error:
        sls.main(["--model", "m", "--adapter", "a", "--data", "unused",
                  "--task", "sst2", "--lora-alpha=" + value])
    assert error.value.code == 2
    assert "--lora-alpha must be finite" in capsys.readouterr().err
