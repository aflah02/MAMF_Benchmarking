from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd
import streamlit as st


APP_TITLE = "MAMF Explorer"

_HARDWARE_RELEASE_ORDER = [
    ("B200", 0),
    ("H200", 1),
    ("H100", 2),
    ("A100", 3),
    ("A10", 4),
    ("T4", 5),
]


def hardware_order_sql(column: str = "hardware") -> str:
    parts = ["CASE"]
    for token, rank in _HARDWARE_RELEASE_ORDER:
        parts.append(f"  WHEN {column} LIKE '%{token}%' THEN {rank}")
    parts.append("  ELSE 999")
    parts.append("END")
    return "\n".join(parts)


def _resolve_db_path(db_path: str) -> str:
    path = Path(db_path)
    if path.is_absolute():
        return str(path)
    return str((Path(__file__).resolve().parent / path).resolve())


@st.cache_resource
def get_db_conn(db_path: str = "matmul.duckdb") -> duckdb.DuckDBPyConnection:
    resolved = _resolve_db_path(db_path)
    try:
        return duckdb.connect(resolved, read_only=True)
    except TypeError:
        return duckdb.connect(resolved)


def require_conn() -> duckdb.DuckDBPyConnection:
    conn = st.session_state.get("conn")
    if conn is None:
        st.error("Database connection not found. Open the home page to initialize the app.")
        st.stop()
    return conn


@st.cache_data(show_spinner=False, hash_funcs={duckdb.DuckDBPyConnection: lambda _: "duckdb_conn"})
def get_distinct_values(conn: duckdb.DuckDBPyConnection, column: str) -> list[str]:
    if column == "hardware":
        rows = conn.execute(
            f"SELECT DISTINCT hardware FROM matmul_results ORDER BY {hardware_order_sql('hardware')}, hardware"
        ).fetchall()
    else:
        rows = conn.execute(f"SELECT DISTINCT {column} FROM matmul_results ORDER BY {column}").fetchall()
    return [row[0] for row in rows]


@st.cache_data(show_spinner=False, hash_funcs={duckdb.DuckDBPyConnection: lambda _: "duckdb_conn"})
def get_db_stats(conn: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    total_rows, hardware_count, dtype_count = conn.execute(
        """
        SELECT
          COUNT(*) AS total_rows,
          COUNT(DISTINCT hardware) AS hardware_count,
          COUNT(DISTINCT dtype) AS dtype_count
        FROM matmul_results
        """
    ).fetchone()
    return {"total_rows": total_rows, "hardware_count": hardware_count, "dtype_count": dtype_count}


@st.cache_data(show_spinner=False, hash_funcs={duckdb.DuckDBPyConnection: lambda _: "duckdb_conn"})
def get_hardware_coverage(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    query = (
        """
        SELECT
          hardware,
          COUNT(DISTINCT dtype) AS dtypes,
          COUNT(DISTINCT (m, n, k)) AS distinct_shapes,
          MAX(max_tflops) AS peak_tflops,
          MAX(m) AS max_m,
          MAX(n) AS max_n,
          MAX(k) AS max_k,
          arg_max(m, (m, n, k)) AS maxshape_m,
          arg_max(n, (m, n, k)) AS maxshape_n,
          arg_max(k, (m, n, k)) AS maxshape_k,
          arg_max(m, (max_tflops, m, n, k)) FILTER (WHERE max_tflops IS NOT NULL) AS peakshape_m,
          arg_max(n, (max_tflops, m, n, k)) FILTER (WHERE max_tflops IS NOT NULL) AS peakshape_n,
          arg_max(k, (max_tflops, m, n, k)) FILTER (WHERE max_tflops IS NOT NULL) AS peakshape_k
        FROM matmul_results
        GROUP BY hardware
        ORDER BY """
        + hardware_order_sql("hardware")
        + """, hardware
        """
    )
    df = conn.execute(query).df()
    if df.empty:
        return df

    df["max_shape_by_mnk"] = (
        df["maxshape_m"].astype(str) + "×" + df["maxshape_n"].astype(str) + "×" + df["maxshape_k"].astype(str)
    )
    has_peak_shape = df["peakshape_m"].notna() & df["peakshape_n"].notna() & df["peakshape_k"].notna()
    df["peak_shape"] = ""
    df.loc[has_peak_shape, "peak_shape"] = (
        df.loc[has_peak_shape, "peakshape_m"].astype(int).astype(str)
        + "×"
        + df.loc[has_peak_shape, "peakshape_n"].astype(int).astype(str)
        + "×"
        + df.loc[has_peak_shape, "peakshape_k"].astype(int).astype(str)
    )
    return df
