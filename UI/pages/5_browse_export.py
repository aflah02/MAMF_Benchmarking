import duckdb
import pandas as pd
import streamlit as st

from ui_common import get_distinct_values, hardware_order_sql, require_conn


def _build_in_clause(column: str, values: list[str]) -> tuple[str, list[str]]:
    if not values:
        return "", []
    placeholders = ", ".join(["?"] * len(values))
    return f" AND {column} IN ({placeholders})", values


@st.cache_data(show_spinner=False, hash_funcs={duckdb.DuckDBPyConnection: lambda _: "duckdb_conn"})
def _query_rows(
    conn: duckdb.DuckDBPyConnection,
    dtype_values: list[str],
    hardware_values: list[str],
    m: int | None,
    n: int | None,
    k: int | None,
    order_by: str,
    limit: int | None,
) -> pd.DataFrame:
    where = "WHERE 1=1"
    params: list[object] = []

    dtype_clause, dtype_params = _build_in_clause("dtype", dtype_values)
    hw_clause, hw_params = _build_in_clause("hardware", hardware_values)
    where += dtype_clause
    where += hw_clause
    params.extend(dtype_params)
    params.extend(hw_params)

    if m is not None:
        where += " AND m=?"
        params.append(int(m))
    if n is not None:
        where += " AND n=?"
        params.append(int(n))
    if k is not None:
        where += " AND k=?"
        params.append(int(k))

    order_map = {
        "Hardware": f"{hardware_order_sql('hardware')}, hardware ASC",
        "Dtype": "dtype ASC",
        "Shape (M,N,K)": "m ASC, n ASC, k ASC",
        "Mean TFLOPS (desc)": "mean_tflops DESC",
        "Median TFLOPS (desc)": "median_tflops DESC",
        "Max TFLOPS (desc)": "max_tflops DESC",
    }
    order_sql = order_map.get(order_by, "hardware ASC")
    limit_sql = f" LIMIT {int(limit)}" if limit is not None else ""

    query = f"""
    SELECT hardware, dtype, m, n, k, mean_tflops, median_tflops, max_tflops
    FROM matmul_results
    {where}
    ORDER BY {order_sql}
    {limit_sql}
    """
    return pd.DataFrame(
        conn.execute(query, params).fetchall(),
        columns=["hardware", "dtype", "m", "n", "k", "mean_tflops", "median_tflops", "max_tflops"],
    )


st.title("Browse & Export")
st.caption("Filter the dataset and export matching rows as CSV.")

conn = require_conn()
dtype_list = get_distinct_values(conn, "dtype")
hardware_list = get_distinct_values(conn, "hardware")

with st.sidebar:
    st.markdown("### Filters")
    with st.form("browse_export_form"):
        dtype_values = st.multiselect("Dtype", dtype_list, default=dtype_list, key="browse_dtype")
        hardware_values = st.multiselect("Hardware", hardware_list, default=hardware_list, key="browse_hw")

        st.markdown("### Optional Exact Shape")
        col1, col2, col3 = st.columns(3)
        with col1:
            m = st.number_input("M", min_value=0, value=0, step=256, key="browse_m")
        with col2:
            n = st.number_input("N", min_value=0, value=0, step=256, key="browse_n")
        with col3:
            k = st.number_input("K", min_value=0, value=0, step=256, key="browse_k")

        st.markdown("### Output")
        order_by = st.selectbox(
            "Order by",
            [
                "Hardware",
                "Dtype",
                "Shape (M,N,K)",
                "Mean TFLOPS (desc)",
                "Median TFLOPS (desc)",
                "Max TFLOPS (desc)",
            ],
            index=3,
            key="browse_order",
        )
        limit = st.number_input(
            "Row limit (0 = no limit)",
            min_value=0,
            max_value=500_000,
            value=50_000,
            step=5_000,
            key="browse_limit",
        )
        submitted = st.form_submit_button("Run")

if submitted:
    df = _query_rows(
        conn,
        dtype_values=dtype_values,
        hardware_values=hardware_values,
        m=int(m) if m else None,
        n=int(n) if n else None,
        k=int(k) if k else None,
        order_by=order_by,
        limit=None if int(limit) == 0 else int(limit),
    )
    st.session_state["browse_export_last"] = {
        "dtype_values": dtype_values,
        "hardware_values": hardware_values,
        "m": int(m) if m else None,
        "n": int(n) if n else None,
        "k": int(k) if k else None,
        "order_by": order_by,
        "limit": int(limit),
        "df": df,
    }

state = st.session_state.get("browse_export_last")
if state is None:
    st.info("Set filters in the sidebar, then click **Run**.")
    st.stop()

df = state["df"]
if df.empty:
    st.warning("No rows found for this filter set.")
    st.stop()

st.caption(f"Rows: {len(df):,}")
st.dataframe(df, use_container_width=True, height=560)

st.download_button(
    "Download CSV",
    df.to_csv(index=False).encode("utf-8"),
    file_name="matmul_results_filtered.csv",
    mime="text/csv",
)
