import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

from ui_common import get_distinct_values, require_conn


def _build_in_clause(values: list[str]) -> tuple[str, list[str]]:
    if not values:
        return "", []
    placeholders = ", ".join(["?"] * len(values))
    return f" AND hardware IN ({placeholders})", values


def _default_index(values: list[int], preferred: int) -> int:
    try:
        return values.index(preferred)
    except ValueError:
        return 0


@st.cache_data(show_spinner=False, hash_funcs={duckdb.DuckDBPyConnection: lambda _: "duckdb_conn"})
def _get_distinct_dim_values(conn: duckdb.DuckDBPyConnection, dim: str, dtype: str) -> list[int]:
    rows = conn.execute(f"SELECT DISTINCT {dim} FROM matmul_results WHERE dtype=? ORDER BY {dim}", (dtype,)).fetchall()
    return [int(r[0]) for r in rows]


st.title("Scaling Curves")
st.caption("Sweep one of M/N/K (hold the others fixed) and compare TFLOPS across hardware.")

conn = require_conn()
dtype_list = get_distinct_values(conn, "dtype")
hardware_list = get_distinct_values(conn, "hardware")

with st.sidebar:
    st.markdown("### Filters")
    with st.form("scaling_form"):
        dtype = st.selectbox("Dtype", dtype_list, key="scaling_dtype")
        metric = st.selectbox(
            "Metric",
            [
                ("mean_tflops", "Mean TFLOPS"),
                ("median_tflops", "Median TFLOPS"),
                ("max_tflops", "Max TFLOPS"),
            ],
            format_func=lambda x: x[1],
            key="scaling_metric",
        )[0]
        selected_hardware = st.multiselect(
            "Hardware",
            hardware_list,
            default=hardware_list[:2] if len(hardware_list) >= 2 else hardware_list[:1],
            key="scaling_hardware",
        )
        sweep_dim = st.selectbox("Sweep", ["M", "N", "K"], key="scaling_sweep")

        st.markdown("### Fixed Dimensions")
        m_values = _get_distinct_dim_values(conn, "m", dtype)
        n_values = _get_distinct_dim_values(conn, "n", dtype)
        k_values = _get_distinct_dim_values(conn, "k", dtype)

        if not m_values or not n_values or not k_values:
            st.error("No (M, N, K) values found for this dtype.")
            st.stop()

        m_fix = st.selectbox("Fixed M", m_values, index=_default_index(m_values, 4096), key="scaling_mfix")
        n_fix = st.selectbox("Fixed N", n_values, index=_default_index(n_values, 4096), key="scaling_nfix")
        k_fix = st.selectbox("Fixed K", k_values, index=_default_index(k_values, 4096), key="scaling_kfix")

        log_x = st.checkbox("Log X", value=False, key="scaling_logx")
        log_y = st.checkbox("Log Y", value=False, key="scaling_logy")

        submitted = st.form_submit_button("Run")

if submitted:
    in_clause, params = _build_in_clause(selected_hardware)

    fixed = {"M": int(m_fix), "N": int(n_fix), "K": int(k_fix)}
    fixed_clause = ""
    fixed_params: list[object] = []
    for dim in ["M", "N", "K"]:
        if dim == sweep_dim:
            continue
        fixed_clause += f" AND {dim.lower()}=?"
        fixed_params.append(fixed[dim])

    query = f"""
    SELECT hardware, m, n, k, {metric} AS tflops
    FROM matmul_results
    WHERE dtype=? {in_clause} {fixed_clause}
    ORDER BY {sweep_dim.lower()} ASC, hardware ASC
    """
    sql_params: list[object] = [dtype, *params, *fixed_params]
    df = pd.DataFrame(
        conn.execute(query, sql_params).fetchall(),
        columns=["Hardware", "M", "N", "K", "TFLOPS"],
    )
    st.session_state["scaling_last"] = {
        "dtype": dtype,
        "metric": metric,
        "hardware": selected_hardware,
        "sweep_dim": sweep_dim,
        "m_fix": int(m_fix),
        "n_fix": int(n_fix),
        "k_fix": int(k_fix),
        "log_x": log_x,
        "log_y": log_y,
        "df": df,
    }

state = st.session_state.get("scaling_last")
if state is None:
    st.info("Set filters in the sidebar, then click **Run**.")
    st.stop()

df = state["df"]
if df.empty:
    st.warning("No rows found for this filter set.")
    st.stop()

sweep_dim = state["sweep_dim"]
df = df.copy()
df["Sweep"] = df[sweep_dim].astype(int)
df["Hardware"] = pd.Categorical(df["Hardware"], categories=state["hardware"], ordered=True)
df = df.sort_values(["Sweep", "Hardware"])

st.caption(
    f"dtype={state['dtype']} | metric={state['metric']} | sweep={sweep_dim} | "
    f"fixed: M={state['m_fix']}, N={state['n_fix']}, K={state['k_fix']} | rows={len(df):,}"
)

missing = sorted(set(state["hardware"]) - set(df["Hardware"].unique()))
if missing:
    st.info(f"Missing data for: {', '.join(missing)}")

tabs = st.tabs(["Chart", "Table"])
with tabs[0]:
    fig = px.line(
        df,
        x="Sweep",
        y="TFLOPS",
        color="Hardware",
        markers=True,
        title=f"Scaling curve (sweep {sweep_dim})",
        labels={"Sweep": sweep_dim, "TFLOPS": "Achieved TFLOPS"},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    fig.update_xaxes(type="log" if state["log_x"] else "linear")
    fig.update_yaxes(type="log" if state["log_y"] else "linear")
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    out = df[["Hardware", "M", "N", "K", "TFLOPS"]]
    st.dataframe(out, use_container_width=True, height=520)
    st.download_button(
        "Download CSV",
        out.to_csv(index=False).encode("utf-8"),
        file_name=(
            f"scaling_{state['dtype']}_{state['metric']}_sweep{state['sweep_dim']}_"
            f"M{state['m_fix']}_N{state['n_fix']}_K{state['k_fix']}.csv"
        ),
        mime="text/csv",
    )
