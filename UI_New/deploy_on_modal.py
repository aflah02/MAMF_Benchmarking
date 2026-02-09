from __future__ import annotations

from pathlib import Path, PurePosixPath

import modal

LOCAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = LOCAL_DIR.parent

DB_CANDIDATES = [
    LOCAL_DIR / "matmul.duckdb",
    REPO_ROOT / "UI" / "matmul.duckdb",
]
DB_PATH = next((p for p in DB_CANDIDATES if p.exists()), DB_CANDIDATES[0])

REMOTE_DIR = PurePosixPath("/root")

REQUIRED_LOCAL_PATHS = [
    LOCAL_DIR / "app.py",
    LOCAL_DIR / "mamf_db.py",
    LOCAL_DIR / "templates",
    LOCAL_DIR / "static",
    DB_PATH,
]

missing = [str(p) for p in REQUIRED_LOCAL_PATHS if not p.exists()]
if missing and modal.is_local():
    raise RuntimeError(f"Missing required UI files/dirs: {', '.join(missing)}")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("fastapi", "uvicorn", "jinja2", "duckdb", "python-multipart")
    .add_local_file(LOCAL_DIR / "app.py", str(REMOTE_DIR / "app.py"))
    .add_local_file(LOCAL_DIR / "mamf_db.py", str(REMOTE_DIR / "mamf_db.py"))
    .add_local_dir(LOCAL_DIR / "templates", str(REMOTE_DIR / "templates"))
    .add_local_dir(LOCAL_DIR / "static", str(REMOTE_DIR / "static"))
    .add_local_file(DB_PATH, str(REMOTE_DIR / "matmul.duckdb"))
    .env({"MAMF_DB_PATH": str(REMOTE_DIR / "matmul.duckdb")})
)

app = modal.App(name="mamf-explorer-new", image=image)


@app.function(timeout=60 * 10, min_containers=1)
@modal.asgi_app()
def web():
    from app import app as fastapi_app

    return fastapi_app
