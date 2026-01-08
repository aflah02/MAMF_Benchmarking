import streamlit as st

from ui_common import APP_TITLE, get_db_stats, get_hardware_coverage, require_conn

st.title(APP_TITLE)
st.caption("Explore Maximum Achievable Matmul FLOPS (MAMF) across shapes, dtypes, and hardware.")

conn = require_conn()
stats = get_db_stats(conn)
coverage = get_hardware_coverage(conn)

st.markdown(
    """
### What is MAMF?

**MAMF (Maximum Achievable Matmul FLOPS)** is a practical upper bound on matrix-multiplication throughput for a given
GPU + software stack (PyTorch/CUDA/cuBLAS, etc). The benchmark sweeps many `(M, N, K)` shapes and records the best
achieved TFLOPS for each shape, giving you a map of **where the hardware is fast** and **where performance cliffs are**.

Note: The reported results use a Debian-slim based image from Modal with Python 3.12 and PyTorch 2.9.0. At the moment all runs use BF16 and I plan to add FP8 in the future.

### Why should you care?

- **Capacity planning**: estimate how close your workloads can get to peak throughput for the shapes you actually run.
- **Hardware comparisons**: compare GPUs on the same shapes/dtype (not just a single cherry-picked benchmark).
- **Performance debugging**: find regimes where you are bandwidth/launch-bound or hitting kernel selection issues.
- **Regression tracking**: compare results across PyTorch/CUDA versions to spot wins/losses [WIP].

### How is MAMF measured?

This project uses the `mamf_finder.py` script originally published by Stas Bekman in the
`ml-engineering` repository.
"""
)

left, mid, right = st.columns(3)
left.metric("Rows", f"{stats['total_rows']:,}")
mid.metric("GPUs", f"{stats['hardware_count']:,}")
right.metric("Dtypes", f"{stats['dtype_count']:,}")

st.subheader("Coverage & Broad Statistics:")
st.text("I kicked off runs where each of M, N, K ranged from 0 to 20,480 in steps of 256 however not all runs are complete yet and hence coverage varies by GPU.")
st.caption("Max shape uses lexicographic ordering by (M desc, N desc, K desc); some runs may be partial per GPU.")
if coverage.empty:
    st.info("No coverage stats available.")
else:
    view = coverage[["hardware", "distinct_shapes", "max_shape_by_mnk", "peak_tflops", "peak_shape"]].copy()
    # Streamlit's dataframe widget right-aligns numeric columns by default; cast to strings for uniform left alignment.
    view["distinct_shapes"] = view["distinct_shapes"].map(lambda x: f"{int(x):,}")
    view["peak_tflops"] = view["peak_tflops"].map(lambda x: f"{float(x):.1f}")
    st.dataframe(
        view.rename(
            columns={
                "hardware": "Hardware",
                "distinct_shapes": "Distinct shapes",
                "max_shape_by_mnk": "Max shape tried (M,N,K)",
                "peak_tflops": "Peak TFLOPS observed",
                "peak_shape": "Peak TFLOPS shape (M,N,K)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

st.subheader("Pages")
st.write("- Lookup Shape: find TFLOPS for a single (M, N, K) on one GPU/dtype.")
st.write("- Fast & Slow Shapes: discover the best (and optionally worst) shapes for a fixed K.")
st.write("- Scaling Curves: sweep M/N/K (hold others fixed) and compare across hardware.")
st.write("- Compare Hardware: compare a single shape across GPUs (same dtype).")
st.write("- Browse & Export: filter the full table and download CSV.")

# with st.sidebar:
#     st.markdown("### About")
#     st.write("Use the pages to query the DuckDB dataset and visualize performance.")
#     st.caption("Data source: `UI/matmul.duckdb`")
