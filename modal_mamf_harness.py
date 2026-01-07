from __future__ import annotations

import argparse
import datetime
import shlex
import subprocess
from pathlib import Path

import modal

BASE_IMAGE="debian-slim"
GPU = "H200" # Change to your desired GPU type
PYTHON_VERSION = "3.12" # Change to your desired Python version
TORCH_VERSION = "2.9.0" # Change to your desired PyTorch version

# Picked ranges from here - https://github.com/stas00/ml-engineering/tree/master/compute/accelerator/benchmarks#maximum-achievable-matmul-flops-finder
M_RANGE = "0 20480 256"
N_RANGE = "0 20480 256"
K_RANGE = "0 20480 256"
MAMF_ARGS = f"--m_range {M_RANGE} --n_range {N_RANGE} --k_range {K_RANGE} --num_iterations 100 --num_warmup_iterations 50 --dtype bfloat16"

SUPPORTED_PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]
SUPPORTED_TORCH_VERSIONS = [
    "2.9.0", "2.8.0", "2.7.1", "2.7.0", "2.6.0", "2.5.1", "2.5.0",
    "2.4.1", "2.4.0", "2.3.1", "2.3.0", "2.2.2", "2.2.1", "2.2.0",
    "2.1.2", "2.1.1", "2.1.0", "2.0.1", "2.0.0",
]
SUPPORTED_GPU_TYPES = ["A100-80GB", "B200", "H100", "L40S", "A10", "L4", "T4", "A100-40GB", "H200"]

assert GPU in SUPPORTED_GPU_TYPES
assert PYTHON_VERSION in SUPPORTED_PYTHON_VERSIONS
assert TORCH_VERSION in SUPPORTED_TORCH_VERSIONS

# -----------------------
# Modal app + image
# -----------------------
APP_NAME = f"mamf-finder-{GPU}-py{PYTHON_VERSION}-torch{TORCH_VERSION}-base_img-{BASE_IMAGE}"
  
app = modal.App(APP_NAME)

if BASE_IMAGE == "debian-slim":
  image = (
      modal.Image.debian_slim(python_version=PYTHON_VERSION)
      .apt_install("git")
      .uv_pip_install(f"torch=={TORCH_VERSION}", "numpy", "packaging", "setuptools", gpu=GPU)
  )

# -----------------------
# Volume
# -----------------------
vol = modal.Volume.from_name("vol-matmul-analysis", create_if_missing=True)

# with vol.batch_upload(force=True) as batch:
#     batch.put_file("mamf_finder.py", "mamf_finder.py")

VOLUME_MOUNT = {"/data": vol}

# -----------------------
# GPU worker
# -----------------------
@app.function(
    image=image,
    gpu=GPU,
    volumes=VOLUME_MOUNT,
    timeout=60 * 60 * 24,  # 24 hours
)
def run_mamf(mamf_args: str) -> str:
    """
    Runs mamf_finder.py and returns the output file path.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_path = f"/data/outputs/mamf_{timestamp}_{GPU}_py{PYTHON_VERSION}_torch{TORCH_VERSION}_base_img-{BASE_IMAGE}.json"

    if not Path("/data/outputs").exists():
        Path("/data/outputs").mkdir(parents=True, exist_ok=True)

    cmd = (
        ["python", "/data/mamf_finder.py"]
        + shlex.split(mamf_args)
        + ["--output_file", output_path]
    )

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    return output_path


# -----------------------
# Local entrypoint
# -----------------------
@app.local_entrypoint()
def main():
    output_path = run_mamf.remote(MAMF_ARGS)

    print(f"\n✔ Benchmark completed! Output saved to: {output_path}\n")
