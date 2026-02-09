from __future__ import annotations

import threading
from functools import lru_cache
from typing import Any

import duckdb

_HARDWARE_RELEASE_ORDER: list[tuple[str, int]] = [
    ("B200", 0),
    ("H200", 1),
    ("H100", 2),
    ("A100", 3),
    ("A10", 4),
    ("L40S", 5),
    ("L4", 6),
    ("T4", 7),
]

_ALLOWED_METRICS = {"mean_tflops", "median_tflops", "max_tflops"}
_ALLOWED_DIMS = {"m", "n", "k"}

_thread_local = threading.local()


@lru_cache(maxsize=8)
def matmul_columns(db_path: str) -> frozenset[str]:
    conn = get_conn(db_path)
    rows = conn.execute("PRAGMA table_info('matmul_results')").fetchall()
    return frozenset(str(r[1]) for r in rows)


def has_column(db_path: str, column: str) -> bool:
    return column in matmul_columns(db_path)


def hardware_order_sql(column: str = "hardware") -> str:
    parts = ["CASE"]
    for token, rank in _HARDWARE_RELEASE_ORDER:
        parts.append(f"  WHEN {column} LIKE '%{token}%' THEN {rank}")
    parts.append("  ELSE 999")
    parts.append("END")
    return "\n".join(parts)


def _connect(db_path: str) -> duckdb.DuckDBPyConnection:
    try:
        return duckdb.connect(db_path, read_only=True)
    except TypeError:
        return duckdb.connect(db_path)


def get_conn(db_path: str) -> duckdb.DuckDBPyConnection:
    conn: duckdb.DuckDBPyConnection | None = getattr(_thread_local, "conn", None)
    existing_path: str | None = getattr(_thread_local, "db_path", None)
    if conn is None or existing_path != db_path:
        conn = _connect(db_path)
        _thread_local.conn = conn
        _thread_local.db_path = db_path
    return conn


@lru_cache(maxsize=64)
def distinct_values(db_path: str, column: str) -> tuple[str, ...]:
    conn = get_conn(db_path)
    if column not in matmul_columns(db_path):
        return tuple()
    if column == "hardware":
        rows = conn.execute(
            f"SELECT DISTINCT hardware FROM matmul_results ORDER BY {hardware_order_sql('hardware')}, hardware"
        ).fetchall()
    else:
        rows = conn.execute(f"SELECT DISTINCT {column} FROM matmul_results ORDER BY {column}").fetchall()
    return tuple(row[0] for row in rows)


@lru_cache(maxsize=512)
def distinct_values_for_hardware(db_path: str, column: str, hardware: str) -> tuple[str, ...]:
    conn = get_conn(db_path)
    cols = matmul_columns(db_path)
    if column not in cols or "hardware" not in cols:
        return tuple()
    if not hardware:
        return tuple()
    if column == "hardware":
        return (hardware,)
    rows = conn.execute(
        f"SELECT DISTINCT {column} FROM matmul_results WHERE hardware=? ORDER BY {column}",
        [hardware],
    ).fetchall()
    return tuple(row[0] for row in rows)


@lru_cache(maxsize=8)
def db_stats(db_path: str) -> dict[str, int]:
    conn = get_conn(db_path)
    cols = matmul_columns(db_path)
    select = [
        "COUNT(*) AS total_rows",
        "COUNT(DISTINCT hardware) AS hardware_count",
        "COUNT(DISTINCT dtype) AS dtype_count",
    ]
    if "torch_version" in cols:
        select.append("COUNT(DISTINCT torch_version) AS torch_version_count")
    if "python_version" in cols:
        select.append("COUNT(DISTINCT python_version) AS python_version_count")

    row = conn.execute(f"SELECT {', '.join(select)} FROM matmul_results").fetchone()
    out = {
        "total_rows": int(row[0]),
        "hardware_count": int(row[1]),
        "dtype_count": int(row[2]),
    }
    idx = 3
    if "torch_version" in cols:
        out["torch_version_count"] = int(row[idx])
        idx += 1
    if "python_version" in cols:
        out["python_version_count"] = int(row[idx])
        idx += 1
    return out


@lru_cache(maxsize=8)
def hardware_coverage(db_path: str) -> list[dict[str, Any]]:
    conn = get_conn(db_path)
    table_cols = matmul_columns(db_path)

    extras: list[str] = []
    col_names = ["hardware", "dtypes"]
    if "torch_version" in table_cols:
        extras.append("COUNT(DISTINCT torch_version) AS torch_versions")
        col_names.append("torch_versions")
    if "python_version" in table_cols:
        extras.append("COUNT(DISTINCT python_version) AS python_versions")
        col_names.append("python_versions")

    extras_sql = (",\n          " + ",\n          ".join(extras)) if extras else ""

    query = (
        f"""
        SELECT
          hardware,
          COUNT(DISTINCT dtype) AS dtypes{extras_sql},
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
        ORDER BY {hardware_order_sql('hardware')}, hardware
        """
    )

    col_names.extend(
        [
            "distinct_shapes",
            "peak_tflops",
            "max_m",
            "max_n",
            "max_k",
            "maxshape_m",
            "maxshape_n",
            "maxshape_k",
            "peakshape_m",
            "peakshape_n",
            "peakshape_k",
        ]
    )

    rows = conn.execute(query).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(zip(col_names, row, strict=True))
        item["max_shape_by_mnk"] = f"{item['maxshape_m']}×{item['maxshape_n']}×{item['maxshape_k']}"
        if item["peakshape_m"] is not None and item["peakshape_n"] is not None and item["peakshape_k"] is not None:
            item["peak_shape"] = f"{int(item['peakshape_m'])}×{int(item['peakshape_n'])}×{int(item['peakshape_k'])}"
        else:
            item["peak_shape"] = ""
        out.append(item)
    return out


def lookup_shape(
    db_path: str,
    *,
    hardware: str,
    dtype: str,
    m: int,
    n: int,
    k: int,
    torch_version: str | None = None,
    python_version: str | None = None,
) -> dict[str, float] | None:
    conn = get_conn(db_path)
    cols = matmul_columns(db_path)
    where = "WHERE hardware=? AND dtype=? AND m=? AND n=? AND k=?"
    params: list[object] = [hardware, dtype, int(m), int(n), int(k)]
    if "torch_version" in cols:
        if torch_version is None:
            raise ValueError("torch_version is required for this DB")
        where += " AND torch_version=?"
        params.append(torch_version)
    if "python_version" in cols and python_version is not None:
        where += " AND python_version=?"
        params.append(python_version)
    row = conn.execute(
        f"""
        SELECT mean_tflops, median_tflops, max_tflops
        FROM matmul_results
        {where}
        """,
        params,
    ).fetchone()
    if row is None:
        return None
    mean_tflops, median_tflops, max_tflops = row
    return {
        "mean_tflops": float(mean_tflops),
        "median_tflops": float(median_tflops),
        "max_tflops": float(max_tflops),
    }


def fast_shapes(
    db_path: str,
    *,
    hardware: str,
    dtype: str,
    torch_version: str | None = None,
    python_version: str | None = None,
    k: int,
    limit: int,
    order: str,
) -> list[dict[str, Any]]:
    order_sql = "DESC" if order.upper() == "DESC" else "ASC"
    conn = get_conn(db_path)
    cols = matmul_columns(db_path)
    where = "WHERE hardware=? AND dtype=? AND k=?"
    params: list[object] = [hardware, dtype, int(k)]
    if "torch_version" in cols:
        if torch_version is None:
            raise ValueError("torch_version is required for this DB")
        where += " AND torch_version=?"
        params.append(torch_version)
    if "python_version" in cols and python_version is not None:
        where += " AND python_version=?"
        params.append(python_version)
    rows = conn.execute(
        f"""
        SELECT m, n, mean_tflops
        FROM matmul_results
        {where}
        ORDER BY mean_tflops {order_sql}
        LIMIT ?
        """,
        (*params, int(limit)),
    ).fetchall()
    return [{"m": int(m), "n": int(n), "tflops": float(t)} for (m, n, t) in rows]


def scaling_curve(
    db_path: str,
    *,
    dtype: str,
    metric: str,
    torch_version: str | None = None,
    python_version: str | None = None,
    hardware: list[str],
    sweep_dim: str,
    m_fix: int,
    n_fix: int,
    k_fix: int,
) -> list[dict[str, Any]]:
    metric = metric.strip()
    if metric not in _ALLOWED_METRICS:
        raise ValueError(f"Invalid metric: {metric}")

    sweep_dim = sweep_dim.strip().lower()
    if sweep_dim not in _ALLOWED_DIMS:
        raise ValueError(f"Invalid sweep dim: {sweep_dim}")

    cols = matmul_columns(db_path)
    where = "WHERE dtype=?"
    params: list[object] = [dtype]

    if "torch_version" in cols:
        if torch_version is None:
            raise ValueError("torch_version is required for this DB")
        where += " AND torch_version=?"
        params.append(torch_version)
    if "python_version" in cols and python_version is not None:
        where += " AND python_version=?"
        params.append(python_version)

    if hardware:
        placeholders = ", ".join(["?"] * len(hardware))
        where += f" AND hardware IN ({placeholders})"
        params.extend(hardware)

    fixed = {"m": int(m_fix), "n": int(n_fix), "k": int(k_fix)}
    for dim in ["m", "n", "k"]:
        if dim == sweep_dim:
            continue
        where += f" AND {dim}=?"
        params.append(fixed[dim])

    conn = get_conn(db_path)
    rows = conn.execute(
        f"""
        SELECT hardware, m, n, k, {metric} AS tflops
        FROM matmul_results
        {where}
        ORDER BY {sweep_dim} ASC, {hardware_order_sql('hardware')}, hardware ASC
        """,
        params,
    ).fetchall()
    return [
        {"hardware": str(h), "m": int(m), "n": int(n), "k": int(k), "tflops": float(t)}
        for (h, m, n, k, t) in rows
    ]


def compare_hardware(
    db_path: str,
    *,
    dtype: str,
    torch_version: str | None = None,
    python_version: str | None = None,
    m: int,
    n: int,
    k: int,
) -> list[dict[str, Any]]:
    conn = get_conn(db_path)
    cols = matmul_columns(db_path)
    where = "WHERE dtype=? AND m=? AND n=? AND k=?"
    params: list[object] = [dtype, int(m), int(n), int(k)]
    if "torch_version" in cols:
        if torch_version is None:
            raise ValueError("torch_version is required for this DB")
        where += " AND torch_version=?"
        params.append(torch_version)
    if "python_version" in cols and python_version is not None:
        where += " AND python_version=?"
        params.append(python_version)
    rows = conn.execute(
        f"""
        SELECT hardware, mean_tflops, median_tflops, max_tflops
        FROM matmul_results
        {where}
        ORDER BY {hardware_order_sql('hardware')}, hardware ASC
        """,
        params,
    ).fetchall()
    return [
        {
            "hardware": str(h),
            "mean_tflops": float(mean),
            "median_tflops": float(median),
            "max_tflops": float(max_),
        }
        for (h, mean, median, max_) in rows
    ]


def browse_rows(
    db_path: str,
    *,
    dtype_values: list[str],
    torch_versions: list[str] | None = None,
    python_versions: list[str] | None = None,
    hardware_values: list[str],
    m: int | None,
    n: int | None,
    k: int | None,
    order_by: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    conn = get_conn(db_path)
    cols = matmul_columns(db_path)

    where = "WHERE 1=1"
    params: list[object] = []

    if dtype_values:
        placeholders = ", ".join(["?"] * len(dtype_values))
        where += f" AND dtype IN ({placeholders})"
        params.extend(dtype_values)

    if hardware_values:
        placeholders = ", ".join(["?"] * len(hardware_values))
        where += f" AND hardware IN ({placeholders})"
        params.extend(hardware_values)

    if torch_versions and "torch_version" in cols:
        placeholders = ", ".join(["?"] * len(torch_versions))
        where += f" AND torch_version IN ({placeholders})"
        params.extend(torch_versions)

    if python_versions and "python_version" in cols:
        placeholders = ", ".join(["?"] * len(python_versions))
        where += f" AND python_version IN ({placeholders})"
        params.extend(python_versions)

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
        "hardware": f"{hardware_order_sql('hardware')}, hardware ASC",
        "dtype": "dtype ASC",
        "shape": "m ASC, n ASC, k ASC",
        "mean_desc": "mean_tflops DESC",
        "median_desc": "median_tflops DESC",
        "max_desc": "max_tflops DESC",
    }
    order_sql = order_map.get(order_by, order_map["mean_desc"])
    limit_sql = f" LIMIT {int(limit)}" if limit is not None else ""

    select_cols = ["hardware", "dtype", "m", "n", "k", "mean_tflops", "median_tflops", "max_tflops"]
    if "torch_version" in cols:
        select_cols.insert(1, "torch_version")
    if "python_version" in cols:
        select_cols.insert(1, "python_version")

    rows = conn.execute(
        f"""
        SELECT {', '.join(select_cols)}
        FROM matmul_results
        {where}
        ORDER BY {order_sql}
        {limit_sql}
        """,
        params,
    ).fetchall()

    out: list[dict[str, Any]] = []
    for row in rows:
        rec = dict(zip(select_cols, row, strict=True))
        out.append(
            {
                "hardware": str(rec["hardware"]),
                "python_version": str(rec["python_version"]) if "python_version" in rec else None,
                "torch_version": str(rec["torch_version"]) if "torch_version" in rec else None,
                "dtype": str(rec["dtype"]),
                "m": int(rec["m"]),
                "n": int(rec["n"]),
                "k": int(rec["k"]),
                "mean_tflops": float(rec["mean_tflops"]),
                "median_tflops": float(rec["median_tflops"]),
                "max_tflops": float(rec["max_tflops"]),
            }
        )
    return out


def compare_dtypes(
    db_path: str,
    *,
    hardware: str,
    torch_version: str | None = None,
    python_version: str | None = None,
    m: int,
    n: int,
    k: int,
) -> list[dict[str, Any]]:
    conn = get_conn(db_path)
    cols = matmul_columns(db_path)
    where = "WHERE hardware=? AND m=? AND n=? AND k=?"
    params: list[object] = [hardware, int(m), int(n), int(k)]
    if "torch_version" in cols:
        if torch_version is None:
            raise ValueError("torch_version is required for this DB")
        where += " AND torch_version=?"
        params.append(torch_version)
    if "python_version" in cols and python_version is not None:
        where += " AND python_version=?"
        params.append(python_version)

    rows = conn.execute(
        f"""
        SELECT dtype, mean_tflops, median_tflops, max_tflops
        FROM matmul_results
        {where}
        ORDER BY dtype ASC
        """,
        params,
    ).fetchall()
    return [
        {
            "dtype": str(dtype),
            "mean_tflops": float(mean),
            "median_tflops": float(median),
            "max_tflops": float(max_),
        }
        for (dtype, mean, median, max_) in rows
    ]


def compare_torch_versions(
    db_path: str,
    *,
    hardware: str,
    dtype: str,
    python_version: str | None = None,
    m: int,
    n: int,
    k: int,
) -> list[dict[str, Any]]:
    conn = get_conn(db_path)
    cols = matmul_columns(db_path)
    if "torch_version" not in cols:
        return []

    where = "WHERE hardware=? AND dtype=? AND m=? AND n=? AND k=?"
    params: list[object] = [hardware, dtype, int(m), int(n), int(k)]
    if "python_version" in cols and python_version is not None:
        where += " AND python_version=?"
        params.append(python_version)

    rows = conn.execute(
        f"""
        SELECT torch_version, mean_tflops, median_tflops, max_tflops
        FROM matmul_results
        {where}
        ORDER BY torch_version ASC
        """,
        params,
    ).fetchall()
    return [
        {
            "torch_version": str(tv),
            "mean_tflops": float(mean),
            "median_tflops": float(median),
            "max_tflops": float(max_),
        }
        for (tv, mean, median, max_) in rows
    ]
