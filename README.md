# Benchmark Matmul (MAMF on Modal)

This repo benchmarks **Maximum Achievable Matmul FLOPS (MAMF)** on GPUs using a Modal remote harness.

## What's here

- `modal_mamf_harness.py`: A Modal app that provisions a GPU worker, runs `mamf_finder.py` remotely, and writes results to a Modal `Volume` under `/data/outputs/`.
- `mamf_finder.py`: Vendored benchmark script (see **Attribution**).

## Using `modal_mamf_harness.py`

1. Install and authenticate Modal (see Modal docs), then run:
   - `modal run modal_mamf_harness.py`
2. Tune the run by editing constants at the top of `modal_mamf_harness.py`:
   - `GPU`, `PYTHON_VERSION`, `TORCH_VERSION`
   - `M_RANGE`, `N_RANGE`, `K_RANGE`, and `MAMF_ARGS` (iterations, dtype, etc.)

### Data/outputs

The harness uses a Modal volume named `vol-matmul-analysis` mounted at `/data`, and expects:
- `mamf_finder.py` to be present at `/data/mamf_finder.py`
- outputs to be written to `/data/outputs/`

If you haven't already populated the volume with `mamf_finder.py`, uncomment the `vol.batch_upload(...)` block in `modal_mamf_harness.py` (or mount the file by another Modal-supported mechanism).

## Attribution

`mamf_finder.py` is copied from [Stas Bekman](https://github.com/stas00)'s repo [here](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/benchmarks/mamf-finder.py)