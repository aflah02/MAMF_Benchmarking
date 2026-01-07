import pandas as pd
import plotly.express as px
import streamlit as st

from ui_common import get_distinct_values, require_conn

st.title("Fast Shapes")
st.caption("Explore the highest-throughput shapes for a selected GPU/dtype (fixed K).")

conn = require_conn()
hardware_list = get_distinct_values(conn, "hardware")
dtype_list = get_distinct_values(conn, "dtype")

with st.sidebar:
    st.markdown("### Filters")
    with st.form("fast_shapes_form"):
        hardware = st.selectbox("Hardware", hardware_list, key="fast_hardware")
        dtype = st.selectbox("Dtype", dtype_list, key="fast_dtype")
        k_fix = st.number_input("Fixed K", min_value=1, value=4096, step=256, key="fast_k")
        top_n = st.number_input("Top N", min_value=10, max_value=5000, value=500, step=10, key="fast_top_n")
        submitted = st.form_submit_button("Run")

if submitted:
    query = """
    SELECT m, n, mean_tflops
    FROM matmul_results
    WHERE hardware=? AND dtype=? AND k=?
    ORDER BY mean_tflops DESC
    LIMIT ?
    """
    df = pd.DataFrame(
        conn.execute(query, (hardware, dtype, k_fix, top_n)).fetchall(),
        columns=["M", "N", "TFLOPS"],
    )
    st.session_state["fast_shapes_last"] = {
        "hardware": hardware,
        "dtype": dtype,
        "k": k_fix,
        "top_n": top_n,
        "df": df,
    }

state = st.session_state.get("fast_shapes_last")
if state is None:
    st.info("Set filters in the sidebar, then click **Run**.")
    st.stop()

df = state["df"]
if df.empty:
    st.warning("No shapes found for this filter set.")
    st.stop()

st.caption(f"{state['hardware']} · {state['dtype']} · K={state['k']} · Top {state['top_n']}")

tabs = st.tabs(["Chart", "Table"])
with tabs[0]:
    fig = px.scatter(
        df,
        x="M",
        y="N",
        color="TFLOPS",
        size="TFLOPS",
        color_continuous_scale="Viridis",
        title="Fastest shapes (by mean TFLOPS)",
        labels={"M": "M", "N": "N", "TFLOPS": "Mean TFLOPS"},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.dataframe(df, use_container_width=True, height=420)
    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"fast_shapes_{state['hardware']}_{state['dtype']}_k{state['k']}.csv",
        mime="text/csv",
    )
