from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]

FILES: dict[str, list[str]] = {
    "H100-py3.12-torch2.8.0-bf16": [
        "UI/outputs/mamf_2026-01-17-17-29-24_H100_py3.12_torch2.8.0_base_img-debian-slim_ResumeFrom-17152x12544x10752.txt",
        "UI/outputs/mamf_2026-01-16-17-21-23_H100_py3.12_torch2.8.0_base_img-debian-slim_ResumeFrom-12032x15360x9728.txt",
        "UI/outputs/mamf_2026-01-12-18-37-39_H100_py3.12_torch2.8.0_base_img-debian-slim_ResumeFrom-None.txt",
    ],
    "H100-py3.12-torch2.7.1-bf16": [
        "UI/outputs/mamf_2026-01-15-07-46-04_H100_py3.12_torch2.7.1_base_img-debian-slim_ResumeFrom-16896x19200x17664.txt",
        "UI/outputs/mamf_2026-01-14-05-53-45_H100_py3.12_torch2.7.1_base_img-debian-slim_ResumeFrom-12032x1792x16640.txt",
        "UI/outputs/mamf_2026-01-12-18-39-57_H100_py3.12_torch2.7.1_base_img-debian-slim_ResumeFrom-None.txt",
    ],
    "H100-py3.12-torch2.9.0-fp8": [
        "UI/outputs/mamf_2026-01-12-18-31-39_H100_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-None.txt",
        "UI/outputs/mamf_2026-01-14-05-57-17_H100_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-16128x15616x18688.txt",
    ],
    "H100-py3.12-torch2.9.0-bf16": [
        "UI/outputs/mamf_2026-01-12-18-19-16_H100_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-17152x14336x18944.txt",
        "UI/outputs/mamf_2026-01-07-18-54-14_H100_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-12032x16896x4864.csv",
        "UI/outputs/mamf_2026-01-06-18-39-23_H100_py3.12_torch2.9.0_base_img-debian-slim.json",
    ],
    "H200-py3.12-torch2.9.0-fp8": [
        "UI/outputs/mamf_2026-01-12-18-31-25_H200_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-None.txt",
        "UI/outputs/mamf_2026-01-14-05-55-28_H200_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-16128x10496x7168.txt",
    ],
    "H200-py3.12-torch2.9.0-bf16": [
        "UI/outputs/mamf_2026-01-12-18-27-30_H200_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-12032x7680x17408.txt",
        "UI/outputs/mamf_2026-01-06-20-19-39_H200_py3.12_torch2.9.0_base_img-debian-slim.json",
        "UI/outputs/mamf_2026-01-14-05-58-36_H200_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-16896x19712x10496.txt"
    ],
    "B200-py3.12-torch2.9.0-bf16": [
        "UI/outputs/mamf_2026-01-06-20-15-12_B200_py3.12_torch2.9.0_base_img-debian-slim.json",
        "UI/outputs/mamf_2026-01-12-18-26-13_B200_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-17152x8192x5120.txt",
    ],
    "B200-py3.12-torch2.9.0-fp8": [
        "UI/outputs/mamf_2026-01-12-18-31-23_B200_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-None.txt",
    ],
    "A100-80GB-py3.12-torch2.9.0-bf16": [
        "UI/outputs/mamf_2026-02-07-19-06-08_A100-80GB_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-14336x13312x19456.txt",
        "UI/outputs/mamf_2026-02-05-18-57-13_A100-80GB_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-12288x13568x1280.txt",
        "UI/outputs/mamf_2026-01-14-23-15-04_A100-80GB_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-9728x17152x3584.txt",
        # "UI/outputs/mamf_2026-01-12-18-41-36_A100-80GB_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-6912x9216x17664.txt", - PCIe Version
        # "UI/outputs/mamf_2026-01-06-18-39-31_A100-80GB_py3.12_torch2.9.0_base_img-debian-slim.json", - PCIe Version
        'UI/outputs/mamf_2026-02-08-19-18-48_A100-80GB_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-16128x12032x2304.txt',
        'UI/outputs/mamf_2026-02-09-10-34-12_A100-80GB_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-None.txt',
        "UI/outputs/mamf_2026-02-09-19-26-13_A100-80GB_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-17920x9472x19712.txt",
        "UI/outputs/mamf_2026-02-10-13-09-08_A100-80GB_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-7424x11520x18176.txt",
        "UI/outputs/mamf_2026-02-11-10-24-41_A100-80GB_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-19200x19456x11776.txt"
    ],
    "A100-40GB-py3.12-torch2.9.0-bf16": [
        "UI/outputs/mamf_2026-01-06-20-17-56_A100-40GB_py3.12_torch2.9.0_base_img-debian-slim.json", # Ends at 7168x15360x14592
        "UI/outputs/mamf_2026-01-12-18-42-31_A100-40GB_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-7168x15360x14592.txt", # Ends at 10240x17152x3072
        "UI/outputs/mamf_2026-02-12-06-57-59_A100-40GB_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-10240x17152x3072.txt",
        "UI/outputs/mamf_2026-02-13-14-06-37_A100-40GB_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-12544x20224x7680.txt",
        "UI/outputs/mamf_2026-02-05-18-59-15_A100-40GB_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-12800x5376x15616.txt",
        "UI/outputs/mamf_2026-02-07-19-05-37_A100-40GB_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-14592x19200x3328.txt",
        # Get-Content path\to\file.txt -Tail 10
        # "UI/outputs/mamf_2026-01-14-23-29-38_A100-40GB_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-10240x17152x3072.txt", # A100 80 GB Version Allocated
        
        "UI/outputs/mamf_2026-02-08-19-19-07_A100-40GB_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-16384x17664x6400.txt",
        "UI/outputs/mamf_2026-02-09-19-25-19_A100-40GB_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-17920x19456x13056.txt",
        "UI/outputs/mamf_2026-02-11-10-32-03_A100-40GB_py3.12_torch2.9.0_base_img-debian-slim_ResumeFrom-19456x17664x3328.txt"
    ],
    "A40-py3.12-torch2.9.0-bf16": [
        "UI/local_run_outputs/a40_mamf_bf16",
    ],
    "L40-py3.12-torch2.9.0-bf16": [
        "UI/local_run_outputs/l40_mamf_bf16",
    ],
    "L40-py3.12-torch2.9.0-fp8": [
        "UI/local_run_outputs/l40_mamf_fp8",
    ],
    "V100-py3.12-torch2.9.0-fp16": [
        "UI/local_run_outputs/v100_mamf_fp16",
    ],
}

_KEY_RE = re.compile(
    r"^(?P<hardware>.+)-py(?P<python_version>[^-]+)-torch(?P<torch_version>[^-]+)-(?P<dtype>[^-]+)$"
)


@dataclass(frozen=True)
class LogGroup:
    key: str
    hardware_key: str
    python_version: str
    torch_version: str
    dtype_key: str


def parse_group_key(key: str) -> LogGroup:
    m = _KEY_RE.match(key.strip())
    if not m:
        raise ValueError(f"Invalid group key format: {key}")

    torch_version=m.group("torch_version").removeprefix("torch")   

    if torch_version == '2.9.0' or torch_version == '2.8.0':
        torch_version += '+cu128'
    elif torch_version == '2.7.1':
        torch_version += '+cu126'

    print(f"Parsed torch version: {torch_version} from key: {key}")

    return LogGroup(
        key=key,
        hardware_key=m.group("hardware"),
        python_version=m.group("python_version").removeprefix("py"),
        torch_version=torch_version,
        dtype_key=m.group("dtype"),
    )


def _iter_input_files(items: Iterable[str]) -> Iterable[Path]:
    for item in items:
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        if path.is_dir():
            yield from sorted([p for p in path.rglob("*") if p.is_file()])
        else:
            yield path


def _load_parse_log_file():
    ui_dir = REPO_ROOT / "UI"
    if str(ui_dir) not in sys.path:
        sys.path.insert(0, str(ui_dir))
    import mamf_log_to_duckdb  # type: ignore

    return mamf_log_to_duckdb.parse_log_file


def _ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS matmul_results (
            hardware TEXT NOT NULL,
            python_version TEXT NOT NULL,
            torch_version TEXT NOT NULL,
            dtype TEXT NOT NULL,
            m INTEGER NOT NULL,
            n INTEGER NOT NULL,
            k INTEGER NOT NULL,
            mean_tflops DOUBLE,
            median_tflops DOUBLE,
            max_tflops DOUBLE,
            group_key TEXT,
            source_log TEXT,
            PRIMARY KEY (hardware, python_version, torch_version, dtype, m, n, k)
        )
        """
    )


def build_combined_db(
    *, out_db: Path, keys: list[str] | None = None, dry_run: bool = False
) -> None:
    parse_log_file = _load_parse_log_file()

    selected_keys = keys if keys else sorted(FILES.keys())
    groups = [parse_group_key(k) for k in selected_keys]

    inserted_files = 0
    inserted_rows = 0
    missing: list[str] = []

    if dry_run:
        for group in groups:
            for log_path in _iter_input_files(FILES.get(group.key, [])):
                if not log_path.exists():
                    missing.append(str(log_path))
                    continue

                print(f"[{group.key}] parsing {log_path} ...")
                t0 = time.time()
                df = parse_log_file(str(log_path))
                inserted_files += 1
                inserted_rows += int(len(df))
                print(f"[{group.key}] {len(df):,} rows ({time.time() - t0:.2f}s)")
    else:
        out_db.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing combined DB: {out_db}")
        con = duckdb.connect(str(out_db))
        _ensure_schema(con)

        con.execute("BEGIN")
        try:
            for group in groups:
                for log_path in _iter_input_files(FILES.get(group.key, [])):
                    if not log_path.exists():
                        missing.append(str(log_path))
                        continue

                    print(f"[{group.key}] parsing {log_path} ...")
                    t0 = time.time()
                    df = parse_log_file(str(log_path))
                    df["python_version"] = group.python_version
                    df["torch_version"] = group.torch_version
                    df["group_key"] = group.key
                    df["source_log"] = str(log_path)

                    df = df[
                        [
                            "hardware",
                            "python_version",
                            "torch_version",
                            "dtype",
                            "m",
                            "n",
                            "k",
                            "mean_tflops",
                            "median_tflops",
                            "max_tflops",
                            "group_key",
                            "source_log",
                        ]
                    ]

                    con.register("tmp_df", df)
                    con.execute(
                        """
                        INSERT OR REPLACE INTO matmul_results (
                            hardware,
                            python_version,
                            torch_version,
                            dtype,
                            m,
                            n,
                            k,
                            mean_tflops,
                            median_tflops,
                            max_tflops,
                            group_key,
                            source_log
                        )
                        SELECT
                            hardware,
                            python_version,
                            torch_version,
                            dtype,
                            m,
                            n,
                            k,
                            mean_tflops,
                            median_tflops,
                            max_tflops,
                            group_key,
                            source_log
                        FROM tmp_df
                        """
                    )

                    inserted_files += 1
                    inserted_rows += int(len(df))
                    print(f"[{group.key}] {len(df):,} rows ({time.time() - t0:.2f}s)")
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


def build_split_dbs(
    *, out_dir: Path, keys: list[str] | None = None, dry_run: bool = False
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_keys = keys if keys else sorted(FILES.keys())
    for key in selected_keys:
        group = parse_group_key(key)
        out_db = out_dir / f"{group.key}.duckdb"
        build_combined_db(out_db=out_db, keys=[group.key], dry_run=dry_run)


def main() -> None:
    default_out = Path(__file__).resolve().parent / "matmul.duckdb"

    parser = argparse.ArgumentParser(description="Build DuckDB files from MAMF output logs.")
    parser.add_argument("--out", type=Path, default=default_out, help="Output DuckDB path (combined mode).")
    parser.add_argument("--split-dir", type=Path, default=None, help="If set, write one DB per group key to this dir.")
    parser.add_argument(
        "--keys",
        nargs="*",
        default=None,
        help="Optional subset of FILES keys to build (default: all).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse inputs but do not write to DuckDB.")
    args = parser.parse_args()

    if args.split_dir is not None:
        build_split_dbs(out_dir=args.split_dir, keys=args.keys, dry_run=args.dry_run)
    else:
        build_combined_db(out_db=args.out, keys=args.keys, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
