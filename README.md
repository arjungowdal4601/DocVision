# Vision Indexer

Foundation for a vision-first PDF indexing pipeline.

The current pipeline renders PDF pages to images, runs a LangGraph page-processing loop, writes memory files, writes one page JSON output per page, exports the graph as Mermaid, logs run events, and records tokenomics. Runtime page processing calls GPT-5.4 vision through LangChain.

## Setup

```powershell
pip install -r requirements.txt
```

## Run

```powershell
python -m vision_indexer.main --pdf tests/fixtures/sample.pdf --out runs/test_real --debug
```

Resume an existing run without overwriting completed page outputs:

```powershell
python -m vision_indexer.main --pdf tests/fixtures/sample.pdf --out runs/test_real --debug --resume
```

Force a clean rerun of the known pipeline outputs:

```powershell
python -m vision_indexer.main --pdf tests/fixtures/sample.pdf --out runs/test_real --debug --force
```

For GPT-5.4 vision processing, set `OPENAI_API_KEY` in `.env`:

```powershell
python -m vision_indexer.main --pdf tests/fixtures/sample.pdf --out runs/test_real --debug
```

## Verify

```powershell
pytest -q
```

## Reliability Files

Each run writes detailed operational state to `run_status.json`. Failures write page-level records under `errors/`, and page progress events are appended to `checkpoints/page_checkpoints.csv`.
