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


@lru_cache(maxsize=8)
def hardware_coverage_by_config(db_path: str) -> list[dict[str, Any]]:
    conn = get_conn(db_path)
    cols = matmul_columns(db_path)

    group_cols = ["hardware", "dtype"]
    select_cols = [
        "hardware",
        "dtype",
    ]

    if "torch_version" in cols:
        group_cols.append("torch_version")
        select_cols.append("torch_version")

    query = f"""
    SELECT
      {', '.join(select_cols)},
      COUNT(DISTINCT (m, n, k)) AS distinct_shapes,
      MAX(max_tflops) AS peak_tflops,
      arg_max(m, (max_tflops, m, n, k)) FILTER (WHERE max_tflops IS NOT NULL) AS peak_m,
      arg_max(n, (max_tflops, m, n, k)) FILTER (WHERE max_tflops IS NOT NULL) AS peak_n,
      arg_max(k, (max_tflops, m, n, k)) FILTER (WHERE max_tflops IS NOT NULL) AS peak_k
    FROM matmul_results
    GROUP BY {', '.join(group_cols)}
    ORDER BY {hardware_order_sql('hardware')}, hardware
    """

    rows = conn.execute(query).fetchall()
    out: list[dict[str, Any]] = []

    for row in rows:
        idx = 0
        hardware = str(row[idx])
        idx += 1
        dtype = str(row[idx])
        idx += 1

        torch_version: str | None = None
        if "torch_version" in cols:
            torch_version = str(row[idx])
            idx += 1

        distinct_shapes = int(row[idx])
        idx += 1
        peak_tflops = row[idx]
        idx += 1
        peak_m, peak_n, peak_k = row[idx : idx + 3]

        peak_shape = ""
        if peak_m is not None and peak_n is not None and peak_k is not None:
            peak_shape = f"{int(peak_m)}x{int(peak_n)}x{int(peak_k)}"

        item: dict[str, Any] = {
            "hardware": hardware,
            "dtype": dtype,
            "distinct_shapes": distinct_shapes,
            "peak_tflops": float(peak_tflops) if peak_tflops is not None else None,
            "peak_shape": peak_shape,
        }
        if torch_version is not None:
            item["torch_version"] = torch_version

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


def scaling_curve_configs(
    db_path: str,
    *,
    configs: list[tuple[str, str]],
    metric: str,
    torch_version: str | None = None,
    python_version: str | None = None,
    sweep_dim: str,
    m_fix: int,
    n_fix: int,
    k_fix: int,
) -> list[dict[str, Any]]:
    if not configs:
        return []

    metric = metric.strip()
    if metric not in _ALLOWED_METRICS:
        raise ValueError(f"Invalid metric: {metric}")

    sweep_dim = sweep_dim.strip().lower()
    if sweep_dim not in _ALLOWED_DIMS:
        raise ValueError(f"Invalid sweep dim: {sweep_dim}")

    cols = matmul_columns(db_path)
    params: list[object] = []

    values_rows: list[str] = []
    for hw, dt in configs:
        values_rows.append("(?, ?)")
        params.extend([hw, dt])

    where = "WHERE 1=1"
    if "torch_version" in cols:
        if torch_version is None:
            raise ValueError("torch_version is required for this DB")
        where += " AND r.torch_version=?"
        params.append(torch_version)
    if "python_version" in cols and python_version is not None:
        where += " AND r.python_version=?"
        params.append(python_version)

    fixed = {"m": int(m_fix), "n": int(n_fix), "k": int(k_fix)}
    for dim in ["m", "n", "k"]:
        if dim == sweep_dim:
            continue
        where += f" AND r.{dim}=?"
        params.append(fixed[dim])

    conn = get_conn(db_path)
    rows = conn.execute(
        f"""
        WITH selected(hardware, dtype) AS (
          VALUES {', '.join(values_rows)}
        )
        SELECT r.hardware, r.dtype, r.m, r.n, r.k, r.{metric} AS tflops
        FROM matmul_results r
        JOIN selected s ON r.hardware=s.hardware AND r.dtype=s.dtype
        {where}
        ORDER BY r.{sweep_dim} ASC, {hardware_order_sql('r.hardware')}, r.hardware ASC, r.dtype ASC
        """,
        params,
    ).fetchall()

    out: list[dict[str, Any]] = []
    for (h, dt, m, n, k, t) in rows:
        out.append(
            {
                "hardware": str(h),
                "dtype": str(dt),
                "m": int(m),
                "n": int(n),
                "k": int(k),
                "tflops": float(t),
                "series": f"{h} · {dt}",
            }
        )
    return out


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


def top_shapes(
    db_path: str,
    *,
    hardware: str,
    dtype: str,
    metric: str,
    torch_version: str | None = None,
    python_version: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    metric = metric.strip()
    if metric not in _ALLOWED_METRICS:
        raise ValueError(f"Invalid metric: {metric}")

    conn = get_conn(db_path)
    cols = matmul_columns(db_path)

    where = "WHERE hardware=? AND dtype=?"
    params: list[object] = [hardware, dtype]

    if "torch_version" in cols:
        if torch_version is None:
            raise ValueError("torch_version is required for this DB")
        where += " AND torch_version=?"
        params.append(torch_version)
    if "python_version" in cols and python_version is not None:
        where += " AND python_version=?"
        params.append(python_version)

    limit_val = max(1, min(int(limit), 5000))
    rows = conn.execute(
        f"""
        SELECT m, n, k, mean_tflops, median_tflops, max_tflops
        FROM matmul_results
        {where}
        ORDER BY {metric} DESC
        LIMIT ?
        """,
        (*params, limit_val),
    ).fetchall()

    return [
        {
            "m": int(m),
            "n": int(n),
            "k": int(k),
            "mean_tflops": float(mean),
            "median_tflops": float(median),
            "max_tflops": float(max_),
        }
        for (m, n, k, mean, median, max_) in rows
    ]


def stability_outliers(
    db_path: str,
    *,
    hardware: str,
    dtype: str,
    sort_by: str,
    torch_version: str | None = None,
    python_version: str | None = None,
    min_denominator: float = 1.0,
    limit: int = 200,
) -> list[dict[str, Any]]:
    allowed_sorts = {"max_over_mean", "max_over_median", "max_minus_median"}
    sort_by = sort_by.strip().lower()
    if sort_by not in allowed_sorts:
        raise ValueError(f"Invalid sort_by: {sort_by}")

    conn = get_conn(db_path)
    cols = matmul_columns(db_path)

    where = "WHERE hardware=? AND dtype=?"
    params: list[object] = [hardware, dtype]
    if "torch_version" in cols:
        if torch_version is None:
            raise ValueError("torch_version is required for this DB")
        where += " AND torch_version=?"
        params.append(torch_version)
    if "python_version" in cols and python_version is not None:
        where += " AND python_version=?"
        params.append(python_version)

    where += " AND mean_tflops IS NOT NULL AND median_tflops IS NOT NULL AND max_tflops IS NOT NULL"
    where += " AND mean_tflops > 0 AND median_tflops > 0"
    where += " AND median_tflops >= ?"
    params.append(float(min_denominator))

    limit_val = max(1, min(int(limit), 5000))

    order_sql = {
        "max_over_mean": "max_over_mean DESC",
        "max_over_median": "max_over_median DESC",
        "max_minus_median": "max_minus_median DESC",
    }[sort_by]

    rows = conn.execute(
        f"""
        SELECT
          m,
          n,
          k,
          mean_tflops,
          median_tflops,
          max_tflops,
          (max_tflops / NULLIF(mean_tflops, 0)) AS max_over_mean,
          (max_tflops / NULLIF(median_tflops, 0)) AS max_over_median,
          (max_tflops - median_tflops) AS max_minus_median
        FROM matmul_results
        {where}
        ORDER BY {order_sql}
        LIMIT ?
        """,
        (*params, limit_val),
    ).fetchall()

    out: list[dict[str, Any]] = []
    for (m, n, k, mean, median, max_, mom, mome, dmm) in rows:
        out.append(
            {
                "m": int(m),
                "n": int(n),
                "k": int(k),
                "mean_tflops": float(mean),
                "median_tflops": float(median),
                "max_tflops": float(max_),
                "max_over_mean": float(mom) if mom is not None else None,
                "max_over_median": float(mome) if mome is not None else None,
                "max_minus_median": float(dmm) if dmm is not None else None,
            }
        )
    return out


def fp8_vs_bf16_speedup(
    db_path: str,
    *,
    hardware: str,
    torch_version: str | None,
    metric: str,
    k: int | None,
    python_version: str | None = None,
    histogram_bins: int = 50,
    histogram_min: float = 0.5,
    histogram_max: float = 3.0,
    extremes_limit: int = 30,
) -> dict[str, Any]:
    metric = metric.strip()
    if metric not in _ALLOWED_METRICS:
        raise ValueError(f"Invalid metric: {metric}")

    conn = get_conn(db_path)
    cols = matmul_columns(db_path)

    if "torch_version" in cols and torch_version is None:
        raise ValueError("torch_version is required for this DB")

    join_keys = ["m", "n", "k"]
    if "python_version" in cols:
        join_keys.append("python_version")
    if "torch_version" in cols:
        join_keys.append("torch_version")

    def _subquery(dtype: str) -> tuple[str, list[object]]:
        where = "WHERE hardware=? AND dtype=?"
        params: list[object] = [hardware, dtype]
        if "torch_version" in cols:
            where += " AND torch_version=?"
            params.append(torch_version)
        if "python_version" in cols and python_version is not None:
            where += " AND python_version=?"
            params.append(python_version)
        if k is not None:
            where += " AND k=?"
            params.append(int(k))
        sql = f"SELECT {', '.join(join_keys)}, {metric} AS tflops FROM matmul_results {where}"
        return sql, params

    bf16_sql, bf16_params = _subquery("bfloat16")
    fp8_sql, fp8_params = _subquery("float8_e4m3fn")

    params_base = [*bf16_params, *fp8_params]
    cte = f"""
    WITH
      bf16 AS ({bf16_sql}),
      fp8 AS ({fp8_sql}),
      joined AS (
        SELECT
          fp8.{', fp8.'.join(join_keys)},
          bf16.tflops AS bf16_tflops,
          fp8.tflops AS fp8_tflops,
          (fp8.tflops / NULLIF(bf16.tflops, 0)) AS speedup
        FROM bf16
        JOIN fp8
        USING ({', '.join(join_keys)})
        WHERE bf16.tflops IS NOT NULL AND fp8.tflops IS NOT NULL AND bf16.tflops > 0 AND fp8.tflops > 0
      )
    """

    bf16_rows = int(conn.execute(f"SELECT COUNT(*) FROM ({bf16_sql})", bf16_params).fetchone()[0])
    fp8_rows = int(conn.execute(f"SELECT COUNT(*) FROM ({fp8_sql})", fp8_params).fetchone()[0])

    summary_row = conn.execute(
        cte
        + """
        SELECT
          COUNT(*) AS matched,
          AVG(speedup) AS speedup_mean,
          quantile_cont(speedup, 0.5) AS speedup_median,
          quantile_cont(speedup, 0.1) AS speedup_p10,
          quantile_cont(speedup, 0.9) AS speedup_p90,
          quantile_cont(speedup, 0.99) AS speedup_p99,
          COUNT_IF(speedup >= 1.0) AS speedup_ge1,
          COUNT_IF(speedup < ?) AS below_range,
          COUNT_IF(speedup > ?) AS above_range
        FROM joined
        """,
        [*params_base, float(histogram_min), float(histogram_max)],
    ).fetchone()

    matched = int(summary_row[0])
    summary = {
        "bf16_rows": bf16_rows,
        "fp8_rows": fp8_rows,
        "matched": matched,
        "speedup_mean": float(summary_row[1]) if summary_row[1] is not None else None,
        "speedup_median": float(summary_row[2]) if summary_row[2] is not None else None,
        "speedup_p10": float(summary_row[3]) if summary_row[3] is not None else None,
        "speedup_p90": float(summary_row[4]) if summary_row[4] is not None else None,
        "speedup_p99": float(summary_row[5]) if summary_row[5] is not None else None,
        "speedup_ge1": int(summary_row[6]) if summary_row[6] is not None else 0,
        "below_range": int(summary_row[7]) if summary_row[7] is not None else 0,
        "above_range": int(summary_row[8]) if summary_row[8] is not None else 0,
    }

    bins = max(10, min(int(histogram_bins), 120))
    min_val = float(histogram_min)
    max_val = float(histogram_max)
    span = max(1e-9, max_val - min_val)

    hist_rows = conn.execute(
        cte
        + """
        SELECT
          CAST(
            LEAST(
              GREATEST(
                FLOOR(((LEAST(GREATEST(speedup, ?), ?) - ?) / ?) * ?),
                0
              ),
              ? - 1
            ) AS INTEGER
          ) AS bucket,
          COUNT(*) AS count
        FROM joined
        GROUP BY bucket
        ORDER BY bucket ASC
        """,
        [*params_base, min_val, max_val, min_val, span, bins, bins],
    ).fetchall()

    hist = [{"bucket": int(b), "count": int(c)} for (b, c) in hist_rows]

    extremes_n = max(5, min(int(extremes_limit), 200))
    top_rows = conn.execute(
        cte
        + f"""
        SELECT {', '.join(join_keys)}, bf16_tflops, fp8_tflops, speedup
        FROM joined
        ORDER BY speedup DESC
        LIMIT ?
        """,
        [*params_base, extremes_n],
    ).fetchall()
    bottom_rows = conn.execute(
        cte
        + f"""
        SELECT {', '.join(join_keys)}, bf16_tflops, fp8_tflops, speedup
        FROM joined
        ORDER BY speedup ASC
        LIMIT ?
        """,
        [*params_base, extremes_n],
    ).fetchall()

    def _rows_to_dicts(rows: list[tuple[object, ...]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in rows:
            key_vals = row[: len(join_keys)]
            bf16_t, fp8_t, s = row[len(join_keys) :]
            rec: dict[str, Any] = {}
            for kname, kval in zip(join_keys, key_vals, strict=True):
                rec[kname] = str(kval) if kname in {"python_version", "torch_version"} else int(kval)
            rec["bf16_tflops"] = float(bf16_t)
            rec["fp8_tflops"] = float(fp8_t)
            rec["speedup"] = float(s)
            out.append(rec)
        return out

    return {
        "summary": summary,
        "histogram": {
            "bins": bins,
            "min": min_val,
            "max": max_val,
            "data": hist,
        },
        "top": _rows_to_dicts(top_rows),
        "bottom": _rows_to_dicts(bottom_rows),
    }


def compare_configs_speedup(
    db_path: str,
    *,
    hardware_a: str,
    dtype_a: str,
    torch_version_a: str | None,
    hardware_b: str,
    dtype_b: str,
    torch_version_b: str | None,
    metric: str,
    k: int | None,
    python_version: str | None = None,
    histogram_bins: int = 50,
    histogram_min: float = 0.5,
    histogram_max: float = 3.0,
    extremes_limit: int = 30,
    min_tflops: float = 1.0,
) -> dict[str, Any]:
    metric = metric.strip()
    if metric not in _ALLOWED_METRICS:
        raise ValueError(f"Invalid metric: {metric}")

    conn = get_conn(db_path)
    cols = matmul_columns(db_path)

    if "torch_version" in cols and (torch_version_a is None or torch_version_b is None):
        raise ValueError("torch_version is required for this DB")

    join_keys = ["m", "n", "k"]

    def _subquery(*, hardware: str, dtype: str, torch_version: str | None) -> tuple[str, list[object]]:
        where = "WHERE hardware=? AND dtype=?"
        params: list[object] = [hardware, dtype]
        if "torch_version" in cols:
            where += " AND torch_version=?"
            params.append(torch_version)
        if "python_version" in cols and python_version is not None:
            where += " AND python_version=?"
            params.append(python_version)
        if k is not None:
            where += " AND k=?"
            params.append(int(k))
        sql = f"SELECT {', '.join(join_keys)}, {metric} AS tflops FROM matmul_results {where}"
        return sql, params

    a_sql, a_params = _subquery(hardware=hardware_a, dtype=dtype_a, torch_version=torch_version_a)
    b_sql, b_params = _subquery(hardware=hardware_b, dtype=dtype_b, torch_version=torch_version_b)

    params_base = [*a_params, *b_params]
    params_joined = [*params_base, float(min_tflops), float(min_tflops)]

    cte = f"""
    WITH
      a AS ({a_sql}),
      b AS ({b_sql}),
      joined AS (
        SELECT
          a.{', a.'.join(join_keys)},
          a.tflops AS a_tflops,
          b.tflops AS b_tflops,
          (a.tflops / NULLIF(b.tflops, 0)) AS speedup,
          (a.tflops - b.tflops) AS delta_tflops
        FROM a
        JOIN b
        USING ({', '.join(join_keys)})
        WHERE a.tflops IS NOT NULL
          AND b.tflops IS NOT NULL
          AND a.tflops > 0
          AND b.tflops > 0
          AND a.tflops >= ?
          AND b.tflops >= ?
      )
    """

    a_rows = int(conn.execute(f"SELECT COUNT(*) FROM ({a_sql})", a_params).fetchone()[0])
    b_rows = int(conn.execute(f"SELECT COUNT(*) FROM ({b_sql})", b_params).fetchone()[0])

    summary_row = conn.execute(
        cte
        + """
        SELECT
          COUNT(*) AS matched,
          AVG(speedup) AS speedup_mean,
          quantile_cont(speedup, 0.5) AS speedup_median,
          quantile_cont(speedup, 0.1) AS speedup_p10,
          quantile_cont(speedup, 0.9) AS speedup_p90,
          quantile_cont(speedup, 0.99) AS speedup_p99,
          COUNT_IF(speedup > 1.0) AS a_faster,
          COUNT_IF(speedup < 1.0) AS b_faster,
          COUNT_IF(speedup < ?) AS below_range,
          COUNT_IF(speedup > ?) AS above_range
        FROM joined
        """,
        [*params_joined, float(histogram_min), float(histogram_max)],
    ).fetchone()

    matched = int(summary_row[0])
    a_faster = int(summary_row[6]) if summary_row[6] is not None else 0
    b_faster = int(summary_row[7]) if summary_row[7] is not None else 0
    summary = {
        "a_rows": a_rows,
        "b_rows": b_rows,
        "matched": matched,
        "a_faster": a_faster,
        "b_faster": b_faster,
        "equal": max(0, matched - a_faster - b_faster),
        "speedup_mean": float(summary_row[1]) if summary_row[1] is not None else None,
        "speedup_median": float(summary_row[2]) if summary_row[2] is not None else None,
        "speedup_p10": float(summary_row[3]) if summary_row[3] is not None else None,
        "speedup_p90": float(summary_row[4]) if summary_row[4] is not None else None,
        "speedup_p99": float(summary_row[5]) if summary_row[5] is not None else None,
        "below_range": int(summary_row[8]) if summary_row[8] is not None else 0,
        "above_range": int(summary_row[9]) if summary_row[9] is not None else 0,
    }

    bins = max(10, min(int(histogram_bins), 120))
    min_val = float(histogram_min)
    max_val = float(histogram_max)
    span = max(1e-9, max_val - min_val)

    hist_rows = conn.execute(
        cte
        + """
        SELECT
          CAST(
            LEAST(
              GREATEST(
                FLOOR(((LEAST(GREATEST(speedup, ?), ?) - ?) / ?) * ?),
                0
              ),
              ? - 1
            ) AS INTEGER
          ) AS bucket,
          COUNT(*) AS count
        FROM joined
        GROUP BY bucket
        ORDER BY bucket ASC
        """,
        [*params_joined, min_val, max_val, min_val, span, bins, bins],
    ).fetchall()

    hist = [{"bucket": int(b), "count": int(c)} for (b, c) in hist_rows]

    extremes_n = max(5, min(int(extremes_limit), 200))

    a_faster_rows = conn.execute(
        cte
        + f"""
        SELECT {', '.join(join_keys)}, a_tflops, b_tflops, speedup, delta_tflops
        FROM joined
        WHERE speedup > 1.0
        ORDER BY speedup DESC
        LIMIT ?
        """,
        [*params_joined, extremes_n],
    ).fetchall()
    b_faster_rows = conn.execute(
        cte
        + f"""
        SELECT {', '.join(join_keys)}, a_tflops, b_tflops, speedup, delta_tflops
        FROM joined
        WHERE speedup < 1.0
        ORDER BY speedup ASC
        LIMIT ?
        """,
        [*params_joined, extremes_n],
    ).fetchall()

    def _rows_to_dicts(rows: list[tuple[object, ...]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in rows:
            key_vals = row[: len(join_keys)]
            a_t, b_t, s, d = row[len(join_keys) :]
            rec: dict[str, Any] = {}
            for kname, kval in zip(join_keys, key_vals, strict=True):
                rec[kname] = int(kval)
            rec["a_tflops"] = float(a_t)
            rec["b_tflops"] = float(b_t)
            rec["speedup"] = float(s)
            rec["delta_tflops"] = float(d)
            out.append(rec)
        return out

    return {
        "summary": summary,
        "histogram": {
            "bins": bins,
            "min": min_val,
            "max": max_val,
            "data": hist,
        },
        "a_faster_rows": _rows_to_dicts(a_faster_rows),
        "b_faster_rows": _rows_to_dicts(b_faster_rows),
    }
