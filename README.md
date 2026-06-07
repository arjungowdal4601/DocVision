# Vision Indexer

Stage 1 foundation for a vision-first PDF indexing pipeline.

The current pipeline renders PDF pages to images, runs a LangGraph mock page-processing loop, writes memory files, writes one page JSON output per page, exports the graph as Mermaid, logs run events, and records dummy tokenomics. Real GPT-5.4 vision processing is intentionally not implemented in Stage 1.

## Setup

```powershell
pip install -r requirements.txt
```

## Run

```powershell
python -m vision_indexer.main --pdf tests/fixtures/sample.pdf --out runs/test_run --debug
```

## Verify

```powershell
pytest -q
```
