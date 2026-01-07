import streamlit as st

from ui_common import APP_TITLE, get_db_stats, require_conn

st.title(APP_TITLE)
st.caption("Explore Maximum Achievable Matmul FLOPS (MAMF) across shapes, dtypes, and hardware.")

conn = require_conn()
stats = get_db_stats(conn)

left, mid, right = st.columns(3)
left.metric("Rows", f"{stats['total_rows']:,}")
mid.metric("GPUs", f"{stats['hardware_count']:,}")
right.metric("Dtypes", f"{stats['dtype_count']:,}")

st.divider()

st.subheader("Pages")
st.write("- Lookup Shape: find TFLOPS for a single (M, N, K) on one GPU/dtype.")
st.write("- Fast Shapes: discover the highest-throughput shapes for a fixed K.")
st.write("- Compare Hardware: compare a single shape across GPUs (same dtype).")

with st.sidebar:
    st.markdown("### About")
    st.write("Use the pages to query the DuckDB dataset and visualize performance.")
    st.caption("Data source: `UI/matmul.duckdb`")

