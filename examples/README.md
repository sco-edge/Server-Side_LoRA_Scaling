# Examples

Small validation files in the format the script reads: one JSON object per line with the task's text field and an integer `label`. The sentences are written for illustration only.

| File | Task | Text field |
|---|---|---|
| `sst2.jsonl` | `sst2` | `sentence` |
| `cr.jsonl` | `cr` | `sentence` |
| `sst5.jsonl` | `sst5` | `sentence` |
| `subj.jsonl` | `subj` | `sentence` |
| `trec.jsonl` | `trec` | `question` |
| `agnews.jsonl` | `agnews` | `text` |
| `custom.jsonl` | custom template | `review` |

```bash
python "ZO-FL Server-Side LoRA Scaling.py" --model facebook/opt-1.3b --adapter aggregated_adapter.pt \
  --data examples/sst2.jsonl --task sst2

python "ZO-FL Server-Side LoRA Scaling.py" --model facebook/opt-1.3b --adapter aggregated_adapter.pt \
  --data examples/custom.jsonl \
  --template "Review: {review} Sentiment:" --label-words " negative" " positive"
```
