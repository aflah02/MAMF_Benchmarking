import streamlit as st

from ui_common import APP_TITLE, get_db_stats, require_conn

st.title(APP_TITLE)
st.caption("Explore Maximum Achievable Matmul FLOPS (MAMF) across shapes, dtypes, and hardware.")

conn = require_conn()
stats = get_db_stats(conn)

st.markdown(
    """
### What is MAMF?

**MAMF (Maximum Achievable Matmul FLOPS)** is a practical upper bound on matrix-multiplication throughput for a given
GPU + software stack (PyTorch/CUDA/cuBLAS, etc). The benchmark sweeps many `(M, N, K)` shapes and records the best
achieved TFLOPS for each shape, giving you a map of **where the hardware is fast** and **where performance cliffs are**.

### Why should you care?

- **Capacity planning**: estimate how close your workloads can get to peak throughput for the shapes you actually run.
- **Hardware comparisons**: compare GPUs on the same shapes/dtype (not just a single cherry-picked benchmark).
- **Performance debugging**: find regimes where you are bandwidth/launch-bound or hitting kernel selection issues.
- **Regression tracking**: compare results across PyTorch/CUDA versions to spot wins/losses.

### Where does the benchmark come from?

This project uses the `mamf_finder.py` script originally published by Stas Bekman in the
`ml-engineering` repository (see the repo root `README.md` for attribution and link). We run it remotely via Modal
and load the results into DuckDB for fast querying in this UI.

### How the data gets here

1. Run the Modal harness: `modal run modal_mamf_harness.py`
2. Download outputs locally: `modal volume get vol-matmul-analysis outputs`
3. Parse a log into DuckDB: `python UI/mamf_log_to_duckdb.py <logfile> UI/matmul.duckdb`

The Streamlit app reads from `UI/matmul.duckdb` (read-only) and exposes a few exploration pages below.
"""
)

left, mid, right = st.columns(3)
left.metric("Rows", f"{stats['total_rows']:,}")
mid.metric("GPUs", f"{stats['hardware_count']:,}")
right.metric("Dtypes", f"{stats['dtype_count']:,}")

st.divider()

st.subheader("Pages")
st.write("- Lookup Shape: find TFLOPS for a single (M, N, K) on one GPU/dtype.")
st.write("- Fast Shapes: discover the highest-throughput shapes for a fixed K.")
st.write("- Scaling Curves: sweep M/N/K (hold others fixed) and compare across hardware.")
st.write("- Compare Hardware: compare a single shape across GPUs (same dtype).")
st.write("- Browse & Export: filter the full table and download CSV.")

with st.sidebar:
    st.markdown("### About")
    st.write("Use the pages to query the DuckDB dataset and visualize performance.")
    st.caption("Data source: `UI/matmul.duckdb`")
