import re
import sys
from pathlib import Path
import duckdb
import pandas as pd
import time

# Regex patterns
LOG_LINE_RE = re.compile(
    r"""
    \|\s+
    (?P<mean>[\d\.]+)\(mean\)\s+
    (?P<median>[\d\.]+)\(median\)\s+
    (?P<max>[\d\.]+)\(max\)\s+
    @\s+
    (?P<m>\d+)x(?P<n>\d+)x(?P<k>\d+)
    """,
    re.VERBOSE,
)

DTYPE_RE = re.compile(r"\*\* Dtype:\s+(.+)")
CUDA_DEVICE_RE = re.compile(r"name='([^']+)'")  # <-- key fix

def parse_log_file(path):
    dtype = None
    hardware = None
    rows = []

    with open(path, "r", errors="ignore") as f:
        for line in f:
            if dtype is None:
                m = DTYPE_RE.search(line)
                if m:
                    dtype = m.group(1).replace("torch.", "")

            if hardware is None and "_CudaDeviceProperties" in line:
                m = CUDA_DEVICE_RE.search(line)
                if m:
                    hardware = m.group(1)

            m = LOG_LINE_RE.search(line)
            if m:
                if hardware is None or dtype is None:
                    raise RuntimeError(
                        f"Failed to parse metadata before data rows in {path}"
                    )

                rows.append(
                    (
                        hardware,
                        dtype,
                        int(m.group("m")),
                        int(m.group("n")),
                        int(m.group("k")),
                        float(m.group("mean")),
                        float(m.group("median")),
                        float(m.group("max")),
                    )
                )

    if not rows:
        raise RuntimeError(f"No benchmark rows found in {path}")

    # Convert to DataFrame
    df = pd.DataFrame(
        rows,
        columns=[
            "hardware",
            "dtype",
            "m",
            "n",
            "k",
            "mean_tflops",
            "median_tflops",
            "max_tflops",
        ],
    )
    return df


def main(log_file, db_file):
    
    log_file = Path(log_file)
    db_file = Path(db_file)

    print(f"Parsing log file: {log_file}")
    s_time = time.time()
    df = parse_log_file(log_file)
    print(f"Parsed {len(df)} rows in {time.time() - s_time:.2f} seconds")

    print(f"Connecting to DuckDB at: {db_file}")
    s_time = time.time()
    con = duckdb.connect(db_file)
    print(f"Connected in {time.time() - s_time:.2f} seconds")

    print("Creating table if it does not exist...")
    s_time = time.time()
    # Create table if it doesn't exist
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS matmul_results (
            hardware TEXT NOT NULL,
            dtype TEXT NOT NULL,
            m INTEGER NOT NULL,
            n INTEGER NOT NULL,
            k INTEGER NOT NULL,
            mean_tflops DOUBLE,
            median_tflops DOUBLE,
            max_tflops DOUBLE,
            PRIMARY KEY (hardware, dtype, m, n, k)
        )
        """
    )
    print(f"Table ready in {time.time() - s_time:.2f} seconds")

    print("Registering DataFrame...")
    s_time = time.time()
    # Register the DataFrame as a temporary table in DuckDB
    con.register("tmp_df", df)
    print(f"Data registered in {time.time() - s_time:.2f} seconds")


    print("Upserting data into matmul_results...")
    s_time = time.time()
    # Upsert all rows at once using a single SQL query
    con.execute(
        """
        INSERT OR REPLACE INTO matmul_results
        SELECT * FROM tmp_df
        """
    )
    print(f"Upserted data in {time.time() - s_time:.2f} seconds")

    print(f"Inserted {len(df)} rows from {log_file}")
    print(f"DuckDB saved at: {db_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python mamf_log_to_duckdb.py <logfile> <output.db>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
