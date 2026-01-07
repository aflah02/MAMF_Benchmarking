import duckdb
import sys

def run_query(con, title, query, expect_empty=False):
    print(f"\n=== {title} ===")
    try:
        res = con.execute(query).fetchall()
        if not res:
            if expect_empty:
                print("OK (no rows)")
            else:
                print("No rows returned")
        else:
            for row in res:
                print(row)
            if expect_empty:
                print("WARNING: expected no rows!")
    except Exception as e:
        print(f"ERROR running query: {e}")


def main(db_path):
    con = duckdb.connect(db_path)

    print(f"\nValidating DuckDB file: {db_path}")
    print("=" * 60)

    # 1. Total row count
    run_query(
        con,
        "Total row count",
        "SELECT COUNT(*) AS total_rows FROM matmul_results"
    )

    # 2. Rows per hardware
    run_query(
        con,
        "Row count per hardware",
        """
        SELECT hardware, COUNT(*) AS rows
        FROM matmul_results
        GROUP BY hardware
        ORDER BY rows DESC
        """
    )

    # 3. Distinct shapes per hardware
    run_query(
        con,
        "Distinct shapes per hardware",
        """
        SELECT hardware, COUNT(DISTINCT (m, n, k)) AS distinct_shapes
        FROM matmul_results
        GROUP BY hardware
        """
    )

    # 4. Global M/N/K ranges
    run_query(
        con,
        "Global M/N/K ranges",
        """
        SELECT
          MIN(m) AS min_m, MAX(m) AS max_m,
          MIN(n) AS min_n, MAX(n) AS max_n,
          MIN(k) AS min_k, MAX(k) AS max_k
        FROM matmul_results
        """
    )

    # 5. NULL integrity check
    run_query(
        con,
        "NULL integrity check",
        """
        SELECT
          COUNT(*) FILTER (WHERE hardware IS NULL) AS null_hardware,
          COUNT(*) FILTER (WHERE dtype IS NULL) AS null_dtype,
          COUNT(*) FILTER (
            WHERE m IS NULL OR n IS NULL OR k IS NULL
          ) AS null_dims
        FROM matmul_results
        """
    )

    # 6. Duplicate primary key check
    run_query(
        con,
        "Duplicate (hardware, dtype, m, n, k) rows",
        """
        SELECT hardware, dtype, m, n, k, COUNT(*) AS cnt
        FROM matmul_results
        GROUP BY hardware, dtype, m, n, k
        HAVING COUNT(*) > 1
        """,
        expect_empty=True
    )

    # 7. Performance sanity check
    run_query(
        con,
        "Performance range per hardware",
        """
        SELECT
          hardware,
          MIN(mean_tflops) AS min_mean,
          MAX(mean_tflops) AS max_mean,
          AVG(mean_tflops) AS avg_mean
        FROM matmul_results
        GROUP BY hardware
        """
    )

    # 8. Coverage confidence check
    run_query(
        con,
        "Coverage check per hardware",
        """
        SELECT
          hardware,
          COUNT(*) AS rows,
          COUNT(DISTINCT m) AS distinct_m,
          COUNT(DISTINCT n) AS distinct_n,
          COUNT(DISTINCT k) AS distinct_k
        FROM matmul_results
        GROUP BY hardware
        """
    )

    # 9. Print all data
    run_query(
        con,
        "All data sample",
        """
        SELECT *
        FROM matmul_results
        ORDER BY hardware, m, n, k;
        """
    )

    print("\nValidation complete.")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_matmul_duckdb.py <matmul.duckdb>")
        sys.exit(1)

    main(sys.argv[1])
