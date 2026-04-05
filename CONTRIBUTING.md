# Contributing to Benchmark Matmul MAMF

Thank you for your interest in contributing benchmark results!

## How Contributions Work

1. **Create your contributor folder** under `logs/` with your GitHub username
2. **Run benchmarks** using `mamf_finder.py` (the repo support running locally/on modal but feel free to add your own preferred method)
3. **Save logs** to your contributor folder
4. **Submit a PR** with your new log files and an updated database

The `make_db.py` script automatically discovers all logs under `logs/**/*.txt`, so your logs will be included in the database when you rebuild it.

## Log File Naming Convention

Log files are automatically named by `mamf_finder.py` with the format:
```
mamf_<timestamp>_<GPU>_<python_version>_torch<torch_version>_base_img-<image>_ResumeFrom-<resume_point>.txt
```

Example:
```
mamf_2026-01-06-18-39-23_H100_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-None.txt
```

## Database Schema

Benchmark results are stored in a DuckDB database with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `hardware` | TEXT | Full GPU name (e.g., "NVIDIA H100") |
| `hardware_normalized` | TEXT | Short name (e.g., "H100") |
| `platform` | TEXT | Linux distribution |
| `torch_version` | TEXT | PyTorch version (e.g., "2.9.0+cu124") |
| `dtype` | TEXT | Data type (e.g., "bfloat16") |
| `m`, `n`, `k` | INTEGER | Matrix dimensions |
| `mean_tflops` | DOUBLE | Mean TFLOPS across iterations |
| `median_tflops` | DOUBLE | Median TFLOPS |
| `max_tflops` | DOUBLE | Max TFLOPS (best result) |
| `source_log` | TEXT | Path to source log file |

Primary key: `(hardware, torch_version, dtype, m, n, k)`

## Hardware Normalization

The `make_db.py` script normalizes common GPU names:
- "NVIDIA H100" → "H100"
- "NVIDIA H200" → "H200"
- "NVIDIA B200" → "B200"
- "NVIDIA A100-80GB" → "A100-80GB"
- etc.

## Questions?

Open an issue on GitHub if you have questions or need help!
