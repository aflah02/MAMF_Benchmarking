from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LOG_GLOB = "logs/**/*.txt"

# Hardware alias mapping (GPU model name from log -> friendly key used in DB)
_HARDWARE_ALIASES: dict[str, str] = {
    "NVIDIA H100": "H100",
    "NVIDIA H200": "H200",
    "NVIDIA B200": "B200",
    "NVIDIA A100": "A100",
    "NVIDIA A100-80GB": "A100-80GB",
    "NVIDIA A100-40GB": "A100-40GB",
    "NVIDIA A40": "A40",
    "NVIDIA L40S": "L40S",
    "NVIDIA L40": "L40",
    "NVIDIA V100": "V100",
    "NVIDIA RTX A6000": "RTX6000",
    "NVIDIA GB10": "GB10",
}


def _normalize_hardware(raw: str | None) -> str:
    if raw is None:
        return "unknown"
    return _HARDWARE_ALIASES.get(raw, raw)


def _load_parse_log_file():
    import mamf_log_to_duckdb  # type: ignore

    return mamf_log_to_duckdb.parse_log_file


def _iter_log_files(glob_pattern: str) -> list[Path]:
    """Find all log files matching glob pattern relative to REPO_ROOT."""
    root = REPO_ROOT
    return sorted(root.glob(glob_pattern))


def _ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS matmul_results (
            hardware TEXT NOT NULL,
            hardware_normalized TEXT NOT NULL,
            platform TEXT NOT NULL,
            torch_version TEXT NOT NULL,
            dtype TEXT NOT NULL,
            m INTEGER NOT NULL,
            n INTEGER NOT NULL,
            k INTEGER NOT NULL,
            mean_tflops DOUBLE,
            median_tflops DOUBLE,
            max_tflops DOUBLE,
            source_log TEXT,
            PRIMARY KEY (hardware, torch_version, dtype, m, n, k)
        )
        """
    )


def build_combined_db(
    *, out_db: Path, glob_pattern: str | None = None, dry_run: bool = False
) -> None:
    parse_log_file = _load_parse_log_file()
    pattern = glob_pattern or DEFAULT_LOG_GLOB
    log_files = _iter_log_files(pattern)

    inserted_files = 0
    inserted_rows = 0
    missing: list[Path] = []

    if dry_run:
        for log_path in log_files:
            if not log_path.exists():
                missing.append(log_path)
                continue
            print(f"parsing {log_path} ...")
            t0 = time.time()
            df = parse_log_file(str(log_path))
            inserted_files += 1
            inserted_rows += int(len(df))
            print(f"  -> {len(df):,} rows from {log_path.name} ({time.time() - t0:.2f}s)")
    else:
        out_db.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing combined DB: {out_db}")
        con = duckdb.connect(str(out_db))
        _ensure_schema(con)

        con.execute("BEGIN")
        try:
            for log_path in log_files:
                if not log_path.exists():
                    missing.append(log_path)
                    continue

                print(f"parsing {log_path} ...")
                t0 = time.time()
                df = parse_log_file(str(log_path))

                df["hardware_normalized"] = df["hardware"].apply(_normalize_hardware)
                df["source_log"] = str(log_path)

                df = df[
                    [
                        "hardware",
                        "hardware_normalized",
                        "platform",
                        "torch_version",
                        "dtype",
                        "m",
                        "n",
                        "k",
                        "mean_tflops",
                        "median_tflops",
                        "max_tflops",
                        "source_log",
                    ]
                ]

                con.register("tmp_df", df)
                con.execute(
                    """
                    INSERT OR REPLACE INTO matmul_results (
                        hardware,
                        hardware_normalized,
                        platform,
                        torch_version,
                        dtype,
                        m,
                        n,
                        k,
                        mean_tflops,
                        median_tflops,
                        max_tflops,
                        source_log
                    )
                    SELECT
                        hardware,
                        hardware_normalized,
                        platform,
                        torch_version,
                        dtype,
                        m,
                        n,
                        k,
                        mean_tflops,
                        median_tflops,
                        max_tflops,
                        source_log
                    FROM tmp_df
                    """
                )

                inserted_files += 1
                inserted_rows += int(len(df))
                print(f"  -> {len(df):,} rows ({time.time() - t0:.2f}s)")
            con.execute("COMMIT")
        except Exception as exc:
            try:
                con.execute("ROLLBACK")
            except duckdb.Error:
                pass
            raise
        finally:
            con.close()

    print(f"Done. Files: {inserted_files:,} | Rows: {inserted_rows:,}")
    if missing:
        print("\nMissing inputs:")
        for p in missing:
            print(f"- {p}")


def main() -> None:
    default_out = Path(__file__).resolve().parent / "matmul.duckdb"

    parser = argparse.ArgumentParser(description="Build DuckDB files from MAMF output logs.")
    parser.add_argument("--out", type=Path, default=default_out, help="Output DuckDB path.")
    parser.add_argument(
        "--glob",
        type=str,
        default=DEFAULT_LOG_GLOB,
        help="Glob pattern for log files (relative to repo root).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse inputs but do not write to DuckDB.")
    args = parser.parse_args()

    build_combined_db(out_db=args.out, glob_pattern=args.glob, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
