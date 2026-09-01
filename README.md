# MM-StanceDet

Reference implementation of `MM-StanceDet`, a retrieval-augmented multi-agent
framework for multimodal stance detection. Given an image, text and target, it
goes through retrieval augmentation, multimodal analysis, reasoned debate, and
self-reflection adjudication (details in `../MM-StanceDet_paper_notes.md`).

## Quick start

The stance dataset is already prepared under `data/stance/`, so run directly:

```bash
pip install -r requirements.txt
python scripts/run.py --limit 8
python scripts/evaluate.py --predictions outputs/predictions.jsonl
```

`scripts/run.py` writes predictions with gold/pred labels and a justification to
`outputs/predictions.jsonl`; `scripts/evaluate.py` reports accuracy and Macro-F1.
Use `--offline` to check the pipeline without any API call.

## Configuration

Model, endpoint and key are set in `config/config.yaml`:

```yaml
llm:
  base_url: "https://api.deepseek.com/chat/completions"
  model: "deepseek-v4-flash-vision-exp"
  api_key: ""            # fill in here, or in config/api_key.txt
  api_key_file: "config/api_key.txt"
```

Put the key in `api_key` or in `config/api_key.txt`. Retrieval top-k and debate
rounds are `retrieval.top_k` and `debate.rounds`.

## Folders

`mm_stancedet/` is the core package, `scripts/` holds the entry points, and
`data/stance/` holds the extracted stance data (annotations + images). To rebuild
the full stance subset from the `UniAffect` archive, run
`python scripts/prepare_data.py --full`.

