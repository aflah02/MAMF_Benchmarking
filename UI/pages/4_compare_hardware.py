import pandas as pd
import plotly.express as px
import streamlit as st

from ui_common import get_distinct_values, require_conn

st.title("Compare Hardware")
st.caption("Compare a single shape across GPUs (same dtype).")

conn = require_conn()
dtype_list = get_distinct_values(conn, "dtype")
hardware_list = get_distinct_values(conn, "hardware")

with st.sidebar:
    st.markdown("### Filters")
    with st.form("compare_form"):
        dtype = st.selectbox("Dtype", dtype_list, key="compare_dtype")
        m = st.number_input("M", min_value=1, value=4096, step=256, key="compare_m")
        n = st.number_input("N", min_value=1, value=4096, step=256, key="compare_n")
        k = st.number_input("K", min_value=1, value=4096, step=256, key="compare_k")
        submitted = st.form_submit_button("Compare")

if submitted:
    query = """
    SELECT hardware, mean_tflops, median_tflops, max_tflops
    FROM matmul_results
    WHERE dtype=? AND m=? AND n=? AND k=?
    """
    df = pd.DataFrame(
        conn.execute(query, (dtype, m, n, k)).fetchall(),
        columns=["Hardware", "Mean TFLOPS", "Median TFLOPS", "Max TFLOPS"],
    )
    if not df.empty:
        df["Hardware"] = pd.Categorical(df["Hardware"], categories=hardware_list, ordered=True)
        df = df.sort_values("Hardware")
    st.session_state["compare_last"] = {"dtype": dtype, "m": m, "n": n, "k": k, "df": df}

state = st.session_state.get("compare_last")
if state is None:
    st.info("Set filters in the sidebar, then click **Compare**.")
    st.stop()

df = state["df"]
st.caption(f"{state['dtype']} · {state['m']}×{state['n']}×{state['k']}")

if df.empty:
    st.warning("Shape not found.")
    st.stop()

tabs = st.tabs(["Chart", "Table"])
with tabs[0]:
    fig = px.bar(
        df,
        x="Hardware",
        y="Mean TFLOPS",
        color="Mean TFLOPS",
        text="Mean TFLOPS",
        title="Mean TFLOPS by GPU",
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), yaxis_title="Mean TFLOPS")
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.dataframe(df, use_container_width=True, height=420)
