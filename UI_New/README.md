# UI_New (Fast UI for MAMF Explorer)

This folder contains a **new, faster UI** for the MAMF DuckDB dataset.

## Tech stack (chosen for speed + deployability on Modal)

- **FastAPI**: low-overhead ASGI server and routing
- **HTMX**: “hackable” interactivity without a SPA build step
- **TailwindCSS (CDN)**: modern styling quickly, minimal code
- **DuckDB**: reads `matmul.duckdb` directly (read-only)
- **Plotly.js (CDN)**: client-side charts (no Python plotting overhead)

## Build DBs from logs

`UI_New/make_db.py` contains a mapping of log files grouped by `hardware × python × torch × dtype` and builds:

- A **combined** DB (default): `UI_New/matmul.duckdb`
- Optional **per-group** DBs via `--split-dir`

Examples (from repo root):

- Build combined DB:
  - `python UI_New/make_db.py --out UI_New/matmul.duckdb`
- Build only a subset of groups:
  - `python UI_New/make_db.py --out UI_New/matmul.duckdb --keys H100-py3.12-torch2.9.0-bf16`
- Build one DB per group:
  - `python UI_New/make_db.py --split-dir UI_New/dbs`

## Run locally

From repo root:

1. Install deps:
   - `python -m pip install -r UI_New/requirements.txt`
2. Start the server:
   - `cd UI_New`
   - `uvicorn app:app --reload --port 8000`
3. Open `http://localhost:8000`

By default it will look for the DB in:
- `MAMF_DB_PATH` (if set), else
- `UI_New/matmul.duckdb`, else
- `UI/matmul.duckdb`

## Deploy on Modal

- `modal deploy UI_New/deploy_on_modal.py`

The deployment bundles `UI_New/matmul.duckdb` (if present) or `UI/matmul.duckdb` into the image and sets
`MAMF_DB_PATH=/root/matmul.duckdb`.
