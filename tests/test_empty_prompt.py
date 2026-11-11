from types import SimpleNamespace

import pytest


class Tokenizer:
    pad_token_id = 0

    def __call__(self, prompt):
        return SimpleNamespace(input_ids=[3] if prompt else [])


@pytest.mark.parametrize("words", [[[1], [2]], [[1], [2, 3]]])
def test_empty_tokenized_prompt_in_batch(sls, words):
    model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device="cpu")]))
    with pytest.raises(ValueError, match="prompt must tokenize"):
        sls._class_scores(model, Tokenizer(), ["ok", ""], words, 10)
