import pandas as pd
import plotly.express as px
import streamlit as st

from ui_common import get_distinct_values, require_conn

st.title("Fast & Slow Shapes")
st.caption("Explore the fastest shapes for a selected GPU/dtype (fixed K), and optionally the slowest shapes too.")

conn = require_conn()
hardware_list = get_distinct_values(conn, "hardware")
dtype_list = get_distinct_values(conn, "dtype")

with st.sidebar:
    st.markdown("### Filters")
    with st.form("fast_shapes_form"):
        hardware = st.selectbox("Hardware", hardware_list, key="fast_hardware")
        dtype = st.selectbox("Dtype", dtype_list, key="fast_dtype")
        k_fix = st.number_input("Fixed K", min_value=1, value=4096, step=256, key="fast_k")
        top_n = st.number_input("N shapes", min_value=10, max_value=5000, value=500, step=10, key="fast_top_n")
        show_slow = st.checkbox(
            "Show slow shapes",
            value=False,
            help="Also fetch the bottom N shapes (by mean TFLOPS) for the same fixed K.",
        )
        submitted = st.form_submit_button("Run")

if submitted:
    def load_shapes(order: str) -> pd.DataFrame:
        query = f"""
        SELECT
          m AS "M",
          n AS "N",
          mean_tflops AS "TFLOPS"
        FROM matmul_results
        WHERE hardware=? AND dtype=? AND k=?
        ORDER BY mean_tflops {order}
        LIMIT ?
        """
        return conn.execute(query, (hardware, dtype, k_fix, top_n)).df()

    df_fast = load_shapes("DESC")
    df_slow = load_shapes("ASC") if show_slow else pd.DataFrame(columns=["M", "N", "TFLOPS"])
    df_fast_csv = df_fast.to_csv(index=False).encode("utf-8") if not df_fast.empty else b""
    df_slow_csv = (
        df_slow.to_csv(index=False).encode("utf-8") if show_slow and not df_slow.empty else b""
    )
    st.session_state["fast_shapes_last"] = {
        "hardware": hardware,
        "dtype": dtype,
        "k": k_fix,
        "top_n": top_n,
        "show_slow": show_slow,
        "df_fast": df_fast,
        "df_slow": df_slow,
        "df_fast_csv": df_fast_csv,
        "df_slow_csv": df_slow_csv,
    }

state = st.session_state.get("fast_shapes_last")
if state is None:
    st.info("Set filters in the sidebar, then click **Run**.")
    st.stop()

df_fast = state.get("df_fast", pd.DataFrame(columns=["M", "N", "TFLOPS"]))
df_slow = state.get("df_slow", pd.DataFrame(columns=["M", "N", "TFLOPS"]))
show_slow = bool(state.get("show_slow", False))

if df_fast.empty and df_slow.empty:
    st.warning("No shapes found for this filter set (hardware/dtype/K).")
    st.stop()

st.caption(f"{state['hardware']} · {state['dtype']} · K={state['k']} · N={state['top_n']}")

tabs = st.tabs(["Chart", "Fast Table", "Slow Table"] if show_slow else ["Chart", "Table"])
with tabs[0]:
    if show_slow:
        left, right = st.columns(2)
        with left:
            st.subheader("Fastest shapes")
            if df_fast.empty:
                st.info("No fast shapes found.")
            else:
                fig_fast = px.scatter(
                    df_fast,
                    x="M",
                    y="N",
                    color="TFLOPS",
                    size="TFLOPS",
                    color_continuous_scale="Viridis",
                    title="Top N by mean TFLOPS",
                    labels={"M": "M", "N": "N", "TFLOPS": "Mean TFLOPS"},
                )
                fig_fast.update_layout(margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_fast, use_container_width=True)

        with right:
            st.subheader("Slowest shapes")
            if df_slow.empty:
                st.info("No slow shapes found.")
            else:
                fig_slow = px.scatter(
                    df_slow,
                    x="M",
                    y="N",
                    color="TFLOPS",
                    size="TFLOPS",
                    color_continuous_scale="Viridis",
                    title="Bottom N by mean TFLOPS",
                    labels={"M": "M", "N": "N", "TFLOPS": "Mean TFLOPS"},
                )
                fig_slow.update_layout(margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_slow, use_container_width=True)
    else:
        if df_fast.empty:
            st.info("No shapes found.")
        else:
            fig = px.scatter(
                df_fast,
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
    if df_fast.empty:
        st.info("No fast shapes found.")
    else:
        st.dataframe(df_fast, use_container_width=True, height=420)
        st.download_button(
            "Download CSV",
            state.get("df_fast_csv", b""),
            file_name=f"shapes_fast_{state['hardware']}_{state['dtype']}_k{state['k']}.csv",
            mime="text/csv",
            disabled=df_fast.empty,
        )

if show_slow:
    with tabs[2]:
        if df_slow.empty:
            st.info("No slow shapes found.")
        else:
            st.dataframe(df_slow, use_container_width=True, height=420)
            st.download_button(
                "Download CSV",
                state.get("df_slow_csv", b""),
                file_name=f"shapes_slow_{state['hardware']}_{state['dtype']}_k{state['k']}.csv",
                mime="text/csv",
                disabled=df_slow.empty,
            )
