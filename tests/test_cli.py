import pytest

BASE = ["--model", "m", "--adapter", "a.pt", "--data", "d.jsonl"]


def _exit_code(sls, args):
    with pytest.raises(SystemExit) as e:
        sls.main(BASE + args)
    return e.value.code


def test_requires_task_or_template(sls):
    assert _exit_code(sls, []) == 2
