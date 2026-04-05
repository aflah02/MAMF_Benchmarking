# Benchmark Matmul MAMF

This repository benchmarks **Maximum Achievable Matmul FLOPS (MAMF)** on GPUs. Results are aggregated into a DuckDB database and visualized via a web UI.

## Repository Structure

```
Benchmark Matmul/
├── mamf_finder.py           # Core benchmark script (vendored from stas00/ml-engineering)
├── modal_mamf_harness.py    # Modal app for running benchmarks on remote GPU clusters
├── logs/                    # All benchmark logs, organized by contributor
│   └── [github_username]/   # One folder per contributor
│       └── mamf_*.txt        # Benchmark output files
└── UI/                     # Web UI for exploring results
    ├── app.py              # FastAPI application
    ├── mamf_db.py          # DuckDB query helpers
    ├── mamf_log_to_duckdb.py  # Parse a single log file into DuckDB
    ├── make_db.py          # Build combined DuckDB from all logs
    ├── matmul.duckdb       # Combined database
    ├── templates/          # Jinja2 HTML templates
    └── static/             # CSS and JavaScript
```

## Contributing Benchmark Results

New contributors are welcome! Here's how to add your benchmark results:

### 1. Create Your Contributor Folder

Create a folder under `logs/` with your GitHub username:

```
logs/[your_github_username]/
```

For example, if your GitHub username is `octocat`, create:
```
logs/octocat/
```

### 2. Run the Benchmark

Run `mamf_finder.py` locally or on Modal (see sections below).

### 3. Save Your Logs

Place your generated `.txt` log files in your contributor folder:
```
logs/[your_github_username]/mamf_*.txt
```

### 4. Update the Database

After adding new logs, rebuild the database:

```bash
cd UI
python make_db.py
```

---

## Running the Benchmark

### Option A: Run Locally (on a machine with GPU)

**Requirements:**
- Python 3.10+
- PyTorch with CUDA support
- numpy, packaging

**Quick start:**
```bash
# Full range benchmark
python mamf_finder.py \
  --m_range 0 20480 256 \
  --n_range 0 20480 256 \
  --k_range 0 20480 256 \
  --num_iterations 100 \
  --num_warmup_iterations 50 \
  --dtype bfloat16 \
  --output_file logs/[your_github_username]/mamf_$(date +'%Y-%m-%d-%H-%M-%S').txt

# Quick test (smaller range)
python mamf_finder.py \
  --m_range 2048 4096 256 \
  --n_range 2048 4096 256 \
  --k_range 2048 4096 256 \
  --num_iterations 100 \
  --num_warmup_iterations 50 \
  --dtype bfloat16 \
  --output_file logs/[your_github_username]/mamf_test.txt
```

**Resuming an interrupted run:**
```bash
python mamf_finder.py \
  --m_range 0 20480 256 \
  --n_range 0 20480 256 \
  --k_range 0 20480 256 \
  --num_iterations 100 \
  --num_warmup_iterations 50 \
  --dtype bfloat16 \
  --resume_from 12544x20224x7680 \
  --output_file logs/[your_github_username]/mamf_resumed.txt
```

**Key arguments:**
- `--m_range`, `--n_range`, `--k_range`: Matrix dimension ranges as `start stop step`
- `--num_iterations`: Number of benchmark iterations per shape (default: 100)
- `--num_warmup_iterations`: Warmup iterations (default: 50)
- `--dtype`: Data type - `bfloat16`, `float16`, `float32`, or `float8_e4m3fn` (default: bfloat16)
- `--resume_from`: Resume after this shape (format: `MxNxK`) for interrupted runs
- `--output_file`: Output log file path

### Option B: Run on Modal (cloud GPU)

**Requirements:**
- Install Modal: `pip install modal`
- Authenticate with Modal: `modal setup`

**Configuration:**

Edit the constants at the top of `modal_mamf_harness.py`:

```python
GPU = "H100"              # GPU type: A100-80GB, B200, H100, H200, L40S, etc.
PYTHON_VERSION = "3.12"   # Python version
TORCH_VERSION = "2.9.0"   # PyTorch version

# Matrix dimension ranges
M_RANGE = "0 20480 256"
N_RANGE = "0 20480 256"
K_RANGE = "0 20480 256"

DTYPE = "bfloat16"        # Data type

# Resume from a specific shape (or None to start fresh)
RESUME_FROM = None
```

**Run the benchmark:**
```bash
modal run modal_mamf_harness.py
```

**Modal volume:** The harness uses a Modal volume named `vol-matmul-analysis` mounted at `/data`. Logs are written to `/data/outputs/` on Modal and must be downloaded to your local `logs/` folder to be included in the database.

**To download logs from Modal volume:**
```bash
modal volume ls vol-matmul-analysis /data/outputs/
modal volume download vol-matmul-analysis /data/outputs/mamf_*.txt --destination logs/[your_github_username]/
```

---

## Building the Database

After adding new logs, rebuild the DuckDB database:

```bash
cd UI
python make_db.py
```

To do a dry run (parse logs but don't write to DB):
```bash
python make_db.py --dry-run
```

---

## Exploring Results

Run the web UI locally:

```bash
cd UI
pip install -r requirements.txt
python app.py
```

Or deploy to Modal:
```bash
modal deploy UI/deploy_on_modal.py
```

---

## Attribution

`mamf_finder.py` is derived from [Stas Bekman's ml-engineering repository](https://github.com/stas00/ml-engineering) with modifications to add resumption support. The original can be found [here](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/benchmarks/mamf-finder.py).
