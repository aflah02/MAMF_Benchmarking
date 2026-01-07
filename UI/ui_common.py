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
    total_rows = conn.execute("SELECT COUNT(*) FROM matmul_results").fetchone()[0]
    hardware_count = conn.execute("SELECT COUNT(DISTINCT hardware) FROM matmul_results").fetchone()[0]
    dtype_count = conn.execute("SELECT COUNT(DISTINCT dtype) FROM matmul_results").fetchone()[0]
    return {"total_rows": total_rows, "hardware_count": hardware_count, "dtype_count": dtype_count}


@st.cache_data(show_spinner=False, hash_funcs={duckdb.DuckDBPyConnection: lambda _: "duckdb_conn"})
def get_hardware_coverage(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    query = """
    WITH base AS (
      SELECT
        hardware,
        COUNT(*) AS rows,
        COUNT(DISTINCT dtype) AS dtypes,
        COUNT(DISTINCT (m, n, k)) AS distinct_shapes,
        MAX(m) AS max_m,
        MAX(n) AS max_n,
        MAX(k) AS max_k
      FROM matmul_results
      GROUP BY hardware
    ),
    maxshape AS (
      SELECT hardware, m AS maxshape_m, n AS maxshape_n, k AS maxshape_k
      FROM (
        SELECT
          hardware,
          m,
          n,
          k,
          ROW_NUMBER() OVER (PARTITION BY hardware ORDER BY m DESC, n DESC, k DESC) AS rn
        FROM matmul_results
      )
      WHERE rn = 1
    )
    SELECT
      base.*,
      maxshape.maxshape_m,
      maxshape.maxshape_n,
      maxshape.maxshape_k
    FROM base
    JOIN maxshape USING (hardware)
    ORDER BY """ + hardware_order_sql("hardware") + """, hardware
    """
    rows = conn.execute(query).fetchall()
    df = pd.DataFrame(
        rows,
        columns=[
            "hardware",
            "rows",
            "dtypes",
            "distinct_shapes",
            "max_m",
            "max_n",
            "max_k",
            "maxshape_m",
            "maxshape_n",
            "maxshape_k",
        ],
    )
    if df.empty:
        return df

    df["max_shape_by_mnk"] = (
        df["maxshape_m"].astype(str) + "×" + df["maxshape_n"].astype(str) + "×" + df["maxshape_k"].astype(str)
    )
    return df
