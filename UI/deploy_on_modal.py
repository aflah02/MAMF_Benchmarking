import subprocess
from pathlib import Path, PurePosixPath
import shlex
import modal

LOCAL_UI_DIR = Path(__file__).resolve().parent
REMOTE_UI_DIR = PurePosixPath("/root")

REQUIRED_LOCAL_PATHS = [
    LOCAL_UI_DIR / "app.py",
    LOCAL_UI_DIR / "home.py",
    LOCAL_UI_DIR / "ui_common.py",
    LOCAL_UI_DIR / "matmul.duckdb",
    LOCAL_UI_DIR / "pages",
]

missing = [str(path) for path in REQUIRED_LOCAL_PATHS if not path.exists()]
if missing and modal.is_local():
    raise RuntimeError(f"Missing required UI files/dirs: {', '.join(missing)}")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install("streamlit", "numpy", "pandas", "duckdb", "plotly")
    .add_local_file(LOCAL_UI_DIR / "app.py", str(REMOTE_UI_DIR / "app.py"))
    .add_local_file(LOCAL_UI_DIR / "home.py", str(REMOTE_UI_DIR / "home.py"))
    .add_local_file(LOCAL_UI_DIR / "ui_common.py", str(REMOTE_UI_DIR / "ui_common.py"))
    .add_local_file(LOCAL_UI_DIR / "matmul.duckdb", str(REMOTE_UI_DIR / "matmul.duckdb"))
    .add_local_dir(LOCAL_UI_DIR / "pages", str(REMOTE_UI_DIR / "pages"))
)

app = modal.App(name="mamf-explorer", image=image)


@app.function(timeout=60 * 10)
@modal.web_server(8000)
def run_streamlit():
    target = shlex.quote(str(REMOTE_UI_DIR / "app.py"))
    cmd = f"streamlit run {target} --server.port 8000 --server.enableCORS=false --server.enableXsrfProtection=false"
    subprocess.Popen(cmd, shell=True)
