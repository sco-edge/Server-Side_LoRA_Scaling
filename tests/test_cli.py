import pytest

BASE = ["--model", "m", "--adapter", "a.pt", "--data", "d.jsonl"]


def _exit_code(sls, args):
    with pytest.raises(SystemExit) as e:
        sls.main(BASE + args)
    return e.value.code


def test_requires_task_or_template(sls):
    assert _exit_code(sls, []) == 2


def test_requires_label_words_with_template(sls):
    assert _exit_code(sls, ["--template", "{sentence} It was"]) == 2


def test_rejects_batch_size_below_one(sls):
    assert _exit_code(sls, ["--task", "sst2", "--batch-size", "0"]) == 2
