from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import duckdb
import streamlit as st


APP_TITLE = "MAMF Benchmark Explorer"

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
    rows = conn.execute(f"SELECT DISTINCT {column} FROM matmul_results ORDER BY {column}").fetchall()
    return [row[0] for row in rows]


@st.cache_data(show_spinner=False, hash_funcs={duckdb.DuckDBPyConnection: lambda _: "duckdb_conn"})
def get_db_stats(conn: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    total_rows = conn.execute("SELECT COUNT(*) FROM matmul_results").fetchone()[0]
    hardware_count = conn.execute("SELECT COUNT(DISTINCT hardware) FROM matmul_results").fetchone()[0]
    dtype_count = conn.execute("SELECT COUNT(DISTINCT dtype) FROM matmul_results").fetchone()[0]
    return {"total_rows": total_rows, "hardware_count": hardware_count, "dtype_count": dtype_count}
