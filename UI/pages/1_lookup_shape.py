import streamlit as st

from ui_common import get_distinct_values, require_conn

st.title("Lookup Shape")
st.caption("Query a single (M, N, K) and get mean / median / max TFLOPS.")

conn = require_conn()
hardware_list = get_distinct_values(conn, "hardware")
dtype_list = get_distinct_values(conn, "dtype")

with st.sidebar:
    st.markdown("### Filters")
    with st.form("lookup_form"):
        hardware = st.selectbox("Hardware", hardware_list, key="lookup_hardware")
        dtype = st.selectbox("Dtype", dtype_list, key="lookup_dtype")
        m = st.number_input("M", min_value=1, value=4096, step=256, key="lookup_m")
        n = st.number_input("N", min_value=1, value=4096, step=256, key="lookup_n")
        k = st.number_input("K", min_value=1, value=4096, step=256, key="lookup_k")
        submitted = st.form_submit_button("Lookup")

if submitted:
    st.session_state["lookup_last_query"] = {"hardware": hardware, "dtype": dtype, "m": m, "n": n, "k": k}
    query = """
    SELECT mean_tflops, median_tflops, max_tflops
    FROM matmul_results
    WHERE hardware=? AND dtype=? AND m=? AND n=? AND k=?
    """
    st.session_state["lookup_last_result"] = conn.execute(query, (hardware, dtype, m, n, k)).fetchone()

last_query = st.session_state.get("lookup_last_query")
last_result = st.session_state.get("lookup_last_result")

if last_query is None:
    st.info("Set filters in the sidebar, then click **Lookup**.")
elif last_result is None:
    st.warning("Shape not found in database.")
else:
    mean_tflops, median_tflops, max_tflops = last_result
    st.caption(
        f"{last_query['hardware']} · {last_query['dtype']} · "
        f"{last_query['m']}×{last_query['n']}×{last_query['k']}"
    )
    a, b, c = st.columns(3)
    a.metric("Mean TFLOPS", f"{mean_tflops:.1f}")
    b.metric("Median TFLOPS", f"{median_tflops:.1f}")
    c.metric("Max TFLOPS", f"{max_tflops:.1f}")
