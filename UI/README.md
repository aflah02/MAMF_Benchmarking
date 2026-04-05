# UI for MAMF Explorer

## Tech stack

- **FastAPI**: low-overhead ASGI server and routing
- **HTMX**: "hackable" interactivity without a SPA build step
- **TailwindCSS (CDN)**: modern styling quickly, minimal code
- **DuckDB**: reads `matmul.duckdb` directly (read-only)
- **Plotly.js (CDN)**: client-side charts (no Python plotting overhead)

## Build DB from logs

Logs are discovered by glob pattern and parsed directly — no mapping files needed. `make_db.py` extracts hardware, torch version, dtype, and platform from the log headers automatically.

```bash
# Build from repo root
python UI/make_db.py --glob "logs/**/*.txt" --out UI/matmul.duckdb

# Dry run (parse without writing DB)
python UI/make_db.py --glob "logs/**/*.txt" --dry-run
```

The schema includes: `hardware`, `hardware_normalized`, `platform`, `torch_version`, `dtype`, `m`, `n`, `k`, `mean_tflops`, `median_tflops`, `max_tflops`, `source_log`.

## Run locally

1. Install deps:
   ```bash
   pip install -r UI/requirements.txt
   ```
2. Start the server:
   ```bash
   cd UI
   uvicorn app:app --reload --port 8000
   ```
3. Open `http://localhost:8000`

The DB path is resolved via:
- `MAMF_DB_PATH` env var (if set), else
- `UI/matmul.duckdb`

## Deploy on Modal

```bash
modal deploy UI/deploy_on_modal.py
```
