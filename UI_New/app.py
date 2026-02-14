from __future__ import annotations

import csv
import io
import json
import os
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

try:
    from .mamf_db import (  # type: ignore
        browse_rows,
        compare_configs_speedup,
        compare_dtypes,
        compare_hardware,
        compare_torch_versions,
        db_stats,
        distinct_values,
        distinct_values_for_hardware,
        fp8_vs_bf16_speedup,
        fast_shapes,
        hardware_coverage,
        hardware_coverage_by_config,
        lookup_shape,
        scaling_curve,
        scaling_curve_configs,
        stability_outliers,
        top_shapes,
    )
except ImportError:
    from mamf_db import (
        browse_rows,
        compare_configs_speedup,
        compare_dtypes,
        compare_hardware,
        compare_torch_versions,
        db_stats,
        distinct_values,
        distinct_values_for_hardware,
        fp8_vs_bf16_speedup,
        fast_shapes,
        hardware_coverage,
        hardware_coverage_by_config,
        lookup_shape,
        scaling_curve,
        scaling_curve_configs,
        stability_outliers,
        top_shapes,
    )

APP_TITLE = os.getenv("MAMF_APP_TITLE", "MAMF Explorer")
APP_SUBTITLE = os.getenv("MAMF_APP_SUBTITLE", "Maximum Achievable Matmul FLOPS")

ALL_HARDWARE_VALUE = "__all__"
ALL_HARDWARE_LABEL = "All GPUs"
BASELINE_TORCH_VERSION = "2.9.0"

DECLARED_PEAK_TFLOPS_NO_SPARSITY: dict[str, dict[str, float]] = {
    "NVIDIA B200": {"bfloat16": 2250.0, "float16": 2250.0, "float8_e4m3fn": 4500.0},
    "NVIDIA H200": {"bfloat16": 989.0, "float16": 989.0, "float8_e4m3fn": 1979.0},
    "NVIDIA H100 80GB HBM3": {"bfloat16": 989.0, "float16": 989.0, "float8_e4m3fn": 1979.0},
    "NVIDIA L40S": {"bfloat16": 362.05, "float16": 362.05, "float8_e4m3fn": 733.0},
    "NVIDIA A100 80GB PCIe": {"bfloat16": 312.0, "float16": 312.0},
    "NVIDIA A100-SXM4-40GB": {"bfloat16": 312.0, "float16": 312.0},
    "NVIDIA A100-SXM4-80GB": {"bfloat16": 312.0, "float16": 312.0},
    "NVIDIA A40": {"bfloat16": 149.7, "float16": 149.7},
    "Tesla V100-PCIE-32GB": {"float16": 112.0},
}


def declared_peak_tflops_no_sparsity(hardware: str, dtype: str) -> float | None:
    per_hw = DECLARED_PEAK_TFLOPS_NO_SPARSITY.get(hardware)
    if per_hw and dtype in per_hw:
        return float(per_hw[dtype])

    tokens: list[tuple[str, dict[str, float]]] = [
        ("B200", {"bfloat16": 2250.0, "float16": 2250.0, "float8_e4m3fn": 4500.0}),
        ("H200", {"bfloat16": 989.0, "float16": 989.0, "float8_e4m3fn": 1979.0}),
        ("H100", {"bfloat16": 989.0, "float16": 989.0, "float8_e4m3fn": 1979.0}),
        ("L40S", {"bfloat16": 362.05, "float16": 362.05, "float8_e4m3fn": 733.0}),
        ("A100", {"bfloat16": 312.0, "float16": 312.0}),
        ("A40", {"bfloat16": 149.7, "float16": 149.7}),
        ("V100", {"float16": 112.0}),
    ]
    for token, peaks in tokens:
        if token in hardware and dtype in peaks:
            return float(peaks[dtype])
    return None


def _safe_json_dumps(obj: Any) -> str:
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return data.replace("<", "\\u003c")


def _version_key(value: str) -> tuple[int, int, int, str]:
    m = re.match(r"^([0-9]+)[.]([0-9]+)[.]([0-9]+)", value.strip())
    if not m:
        return (0, 0, 0, value)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)), value)


def _build_ui_config(db_path: str) -> dict[str, Any]:
    hardware_list = list(distinct_values(db_path, "hardware"))
    all_dtypes = list(distinct_values(db_path, "dtype"))
    all_torch_versions = sorted(list(distinct_values(db_path, "torch_version")), key=_version_key)

    baseline_tv: str | None = None
    if all_torch_versions:
        baseline_tv = BASELINE_TORCH_VERSION if BASELINE_TORCH_VERSION in all_torch_versions else all_torch_versions[-1]

    hardware_meta: dict[str, dict[str, list[str]]] = {}
    for hw in hardware_list:
        dtypes = list(distinct_values_for_hardware(db_path, "dtype", hw))
        torch_versions = sorted(list(distinct_values_for_hardware(db_path, "torch_version", hw)), key=_version_key)
        hardware_meta[hw] = {
            "dtypes": dtypes,
            "torch_versions": torch_versions,
        }

    dtype_policy_by_token: list[tuple[str, list[str]]] = [
        ("H100", ["bfloat16", "float8_e4m3fn"]),
        ("H200", ["bfloat16", "float8_e4m3fn"]),
        ("B200", ["bfloat16", "float8_e4m3fn"]),
        ("L40", ["bfloat16", "float8_e4m3fn"]),
        ("V100", ["float16"]),
        ("A100", ["bfloat16"]),
    ]

    def _apply_dtype_policy(hw: str, available: list[str]) -> list[str]:
        for token, allowed in dtype_policy_by_token:
            if token in hw:
                filtered = [d for d in allowed if d in available]
                return filtered if filtered else available
        return available

    # Policy: only H100 exposes multiple torch versions in the UI.
    h100_union_versions: set[str] = set()
    for hw in hardware_list:
        meta = hardware_meta.get(hw) or {"dtypes": [], "torch_versions": []}
        meta["dtypes"] = _apply_dtype_policy(hw, list(meta.get("dtypes") or []))

        tvs = list(meta.get("torch_versions") or [])
        if "H100" in hw:
            h100_union_versions.update(tvs)
        else:
            if baseline_tv is None:
                meta["torch_versions"] = []
            elif baseline_tv in tvs:
                meta["torch_versions"] = [baseline_tv]
            elif tvs:
                meta["torch_versions"] = [tvs[-1]]
            else:
                meta["torch_versions"] = [baseline_tv]

        hardware_meta[hw] = {"dtypes": meta["dtypes"], "torch_versions": list(meta["torch_versions"] or [])}

    multi_torch_hardware = [
        hw for hw in hardware_list if "H100" in hw and len(hardware_meta.get(hw, {}).get("torch_versions", [])) > 1
    ]

    # In multi-version scenarios (only H100), keep browse torch options tight to avoid empty selections.
    if h100_union_versions:
        all_torch_versions = sorted(h100_union_versions, key=_version_key)

    return {
        "allHardwareValue": ALL_HARDWARE_VALUE,
        "allHardwareLabel": ALL_HARDWARE_LABEL,
        "baselineTorchVersion": baseline_tv,
        "hardwareList": hardware_list,
        "allDtypes": all_dtypes,
        "allTorchVersions": all_torch_versions,
        "hardwareMeta": hardware_meta,
        "multiTorchHardware": multi_torch_hardware,
    }


def _candidate_db_paths() -> list[Path]:
    here = Path(__file__).resolve().parent
    candidates: list[Path] = []

    env = os.getenv("MAMF_DB_PATH")
    if env:
        candidates.append(Path(env))

    candidates.extend(
        [
            here / "matmul.duckdb",
            here.parent / "UI" / "matmul.duckdb",
            Path.cwd() / "UI" / "matmul.duckdb",
            Path("/data/matmul.duckdb"),
        ]
    )

    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        try:
            resolved = path.expanduser().resolve()
        except Exception:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def resolve_db_path() -> str:
    attempted = _candidate_db_paths()
    for path in attempted:
        if path.exists() and path.is_file():
            return str(path)
    attempted_str = "\n".join([f"- {p}" for p in attempted])
    raise FileNotFoundError(
        "Could not find `matmul.duckdb`.\n"
        "Set `MAMF_DB_PATH` or place the DB at one of:\n"
        f"{attempted_str}"
    )


def create_app() -> FastAPI:
    fastapi_app = FastAPI(title=APP_TITLE)

    db_path = resolve_db_path()
    fastapi_app.state.db_path = db_path
    fastapi_app.state.ui_config = _build_ui_config(db_path)

    templates_dir = Path(__file__).resolve().parent / "templates"
    static_dir = Path(__file__).resolve().parent / "static"

    templates = Jinja2Templates(directory=str(templates_dir))
    fastapi_app.state.templates = templates

    fastapi_app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def render(request: Request, template_name: str, context: dict[str, Any]) -> HTMLResponse:
        base = {
            "request": request,
            "app_title": APP_TITLE,
            "app_subtitle": APP_SUBTITLE,
            "ui_config_json": _safe_json_dumps(fastapi_app.state.ui_config),
        }
        return templates.TemplateResponse(template_name, {**base, **context})

    def is_hx(request: Request) -> bool:
        return request.headers.get("hx-request", "").lower() == "true"

    @fastapi_app.get("/health", response_class=PlainTextResponse)
    def health() -> str:
        return "ok"

    @fastapi_app.get("/", response_class=HTMLResponse)
    def home(request: Request, sort_by: str = "default") -> HTMLResponse:
        stats = db_stats(db_path)
        coverage = hardware_coverage(db_path)
        coverage_by_config = hardware_coverage_by_config(db_path)
        ui: dict[str, Any] = fastapi_app.state.ui_config
        hardware_meta: dict[str, Any] = dict(ui.get("hardwareMeta") or {})
        baseline_torch_version: str | None = ui.get("baselineTorchVersion")
        multi_torch_hardware: list[str] = list(ui.get("multiTorchHardware") or [])
        multi_torch_versions: list[str] = []
        if multi_torch_hardware:
            hw = multi_torch_hardware[0]
            multi_torch_versions = list(ui.get("hardwareMeta", {}).get(hw, {}).get("torch_versions") or [])

        dtype_order = {
            "bfloat16": 0,
            "float16": 1,
            "float8_e4m3fn": 2,
        }

        coverage_rows: list[dict[str, Any]] = []
        if coverage_by_config:
            # Keep home coverage aligned with the UI policy (dtype + torch filtering).
            filtered_by_config: list[dict[str, Any]] = []
            for row in coverage_by_config:
                hw = str(row.get("hardware") or "")
                meta = hardware_meta.get(hw) or {}
                allowed_dtypes = set(meta.get("dtypes") or [])
                allowed_tvs = set(meta.get("torch_versions") or [])

                dtype_val = str(row.get("dtype") or "")
                if allowed_dtypes and dtype_val and dtype_val not in allowed_dtypes:
                    continue

                if "torch_version" in row:
                    tv_val = str(row.get("torch_version") or "")
                    if allowed_tvs and tv_val and tv_val not in allowed_tvs:
                        continue

                filtered_by_config.append(row)

            hardware_seen: list[str] = []
            by_hw: dict[str, list[dict[str, Any]]] = {}
            for row in filtered_by_config:
                hw = str(row.get("hardware") or "")
                if hw not in by_hw:
                    by_hw[hw] = []
                    hardware_seen.append(hw)
                by_hw[hw].append(row)

            for hw in hardware_seen:
                rows = by_hw.get(hw, [])
                by_tv: dict[str, list[dict[str, Any]]] = {}
                tv_seen: list[str] = []
                for r in rows:
                    tv = str(r.get("torch_version") or "")
                    if tv not in by_tv:
                        by_tv[tv] = []
                        tv_seen.append(tv)
                    by_tv[tv].append(r)

                tv_seen_sorted = sorted(tv_seen, key=_version_key, reverse=True) if any(tv_seen) else tv_seen

                torch_groups: list[dict[str, Any]] = []
                for tv in tv_seen_sorted:
                    dtype_rows = by_tv.get(tv, [])
                    dtype_rows_sorted = sorted(
                        dtype_rows,
                        key=lambda rr: (dtype_order.get(str(rr.get("dtype") or ""), 999), str(rr.get("dtype") or "")),
                    )
                    torch_groups.append({"torch_version": tv, "rows": dtype_rows_sorted})

                hw_rowspan = sum(len(g["rows"]) for g in torch_groups)
                hw_first = True
                for tv_group in torch_groups:
                    tv = str(tv_group["torch_version"] or "")
                    tv_rows = list(tv_group["rows"])
                    tv_rowspan = len(tv_rows)
                    tv_first = True
                    for r in tv_rows:
                        dtype_val = str(r.get("dtype") or "")
                        declared_peak = declared_peak_tflops_no_sparsity(hw, dtype_val)
                        measured_peak = r.get("peak_tflops")
                        pct_peak: float | None = None
                        if declared_peak and declared_peak > 0 and measured_peak is not None:
                            try:
                                pct_peak = 100.0 * float(measured_peak) / float(declared_peak)
                            except Exception:
                                pct_peak = None
                        coverage_rows.append(
                            {
                                "hardware": hw,
                                "show_hardware": hw_first,
                                "hardware_rowspan": hw_rowspan,
                                "torch_version": tv,
                                "show_torch": tv_first,
                                "torch_rowspan": tv_rowspan,
                                "declared_peak_tflops": declared_peak,
                                "pct_of_declared_peak": pct_peak,
                                **r,
                            }
                        )
                        hw_first = False
                        tv_first = False

        # Apply sorting based on sort_by parameter (removes grouping)
        if sort_by == "peak_flops":
            coverage_rows.sort(key=lambda x: (x.get("peak_tflops") or -1), reverse=True)
        elif sort_by == "peak_flop_pct":
            coverage_rows.sort(key=lambda x: (x.get("pct_of_declared_peak") or -1), reverse=True)
        
        # Update rowspan flags after sorting (every row now stands alone)
        for i, row in enumerate(coverage_rows):
            row["show_hardware"] = True
            row["hardware_rowspan"] = 1
            row["show_torch"] = True
            row["torch_rowspan"] = 1

        return render(
            request,
            "home.html",
            {
                "title": "Home",
                "stats": stats,
                "coverage": coverage,
                "coverage_rows": coverage_rows,
                "baseline_torch_version": baseline_torch_version,
                "multi_torch_hardware": multi_torch_hardware,
                "multi_torch_versions": multi_torch_versions,
                "sort_by": sort_by,
            },
        )

    @fastapi_app.get("/lookup", response_class=HTMLResponse)
    def lookup(
        request: Request,
        hardware: str | None = None,
        torch_version: str | None = None,
        dtype: str | None = None,
        m: int = 4096,
        n: int = 4096,
        k: int = 4096,
    ) -> HTMLResponse:
        ui: dict[str, Any] = fastapi_app.state.ui_config
        hardware_list: list[str] = ui["hardwareList"]
        hardware_meta: dict[str, Any] = ui["hardwareMeta"]
        all_dtypes: list[str] = ui["allDtypes"]
        baseline_tv: str | None = ui["baselineTorchVersion"]

        hardware = hardware if hardware is not None else (hardware_list[0] if hardware_list else "")
        if hardware not in hardware_list and hardware != ALL_HARDWARE_VALUE:
            hardware = hardware_list[0] if hardware_list else ""

        all_mode = hardware == ALL_HARDWARE_VALUE

        if all_mode:
            dtype_list = all_dtypes[:]
            torch_versions = [baseline_tv] if baseline_tv else []
        else:
            dtype_list = list(hardware_meta.get(hardware, {}).get("dtypes") or all_dtypes)
            torch_versions = list(
                hardware_meta.get(hardware, {}).get("torch_versions") or ([baseline_tv] if baseline_tv else [])
            )

        dtype = dtype if dtype is not None and dtype in dtype_list else (dtype_list[0] if dtype_list else "")

        if baseline_tv is None:
            torch_version = None
        elif all_mode:
            torch_version = baseline_tv
        else:
            if torch_version not in torch_versions:
                if baseline_tv in torch_versions:
                    torch_version = baseline_tv
                elif torch_versions:
                    torch_version = torch_versions[-1]
                else:
                    torch_version = baseline_tv

        has_query = "hardware" in request.query_params and "dtype" in request.query_params

        result: dict[str, float] | None = None
        results: list[dict[str, Any]] | None = None
        if has_query and dtype:
            if all_mode:
                results = []
                for hw in hardware_list:
                    r = lookup_shape(
                        db_path,
                        hardware=hw,
                        torch_version=torch_version,
                        dtype=dtype,
                        m=m,
                        n=n,
                        k=k,
                    )
                    results.append({"hardware": hw, "result": r})
            elif hardware:
                result = lookup_shape(
                    db_path,
                    hardware=hardware,
                    torch_version=torch_version,
                    dtype=dtype,
                    m=m,
                    n=n,
                    k=k,
                )

        any_results_found = bool(results) and any(r.get("result") is not None for r in results or [])

        context = {
            "title": "Lookup Shape",
            "hardware_list": hardware_list,
            "torch_versions": torch_versions,
            "dtype_list": dtype_list,
            "hardware": hardware,
            "torch_version": torch_version,
            "dtype": dtype,
            "m": int(m),
            "n": int(n),
            "k": int(k),
            "has_query": has_query,
            "result": result,
            "results": results,
            "all_mode": all_mode,
            "all_hardware_value": ALL_HARDWARE_VALUE,
            "any_results_found": any_results_found,
        }

        if is_hx(request):
            return render(request, "partials/lookup_result.html", context)
        return render(request, "lookup.html", context)

    @fastapi_app.get("/fast-shapes", response_class=HTMLResponse)
    def fast_shapes_page(
        request: Request,
        hardware: str | None = None,
        torch_version: str | None = None,
        dtype: str | None = None,
        k_fix: int = 4096,
        top_n: int = 500,
        show_slow: bool = False,
    ) -> HTMLResponse:
        ui: dict[str, Any] = fastapi_app.state.ui_config
        hardware_list: list[str] = ui["hardwareList"]
        hardware_meta: dict[str, Any] = ui["hardwareMeta"]
        all_dtypes: list[str] = ui["allDtypes"]
        baseline_tv: str | None = ui["baselineTorchVersion"]

        hardware = hardware if hardware is not None else (hardware_list[0] if hardware_list else "")
        if hardware not in hardware_list:
            hardware = hardware_list[0] if hardware_list else ""

        dtype_list = list(hardware_meta.get(hardware, {}).get("dtypes") or all_dtypes)
        torch_versions = list(
            hardware_meta.get(hardware, {}).get("torch_versions") or ([baseline_tv] if baseline_tv else [])
        )

        dtype = dtype if dtype is not None and dtype in dtype_list else (dtype_list[0] if dtype_list else "")

        if baseline_tv is None:
            torch_version = None
        else:
            if torch_version not in torch_versions:
                if baseline_tv in torch_versions:
                    torch_version = baseline_tv
                elif torch_versions:
                    torch_version = torch_versions[-1]
                else:
                    torch_version = baseline_tv

        has_query = "hardware" in request.query_params and "dtype" in request.query_params
        top_n = max(10, min(int(top_n), 5000))

        fast: list[dict[str, Any]] = []
        slow: list[dict[str, Any]] = []
        if has_query and hardware and dtype:
            fast = fast_shapes(
                db_path,
                hardware=hardware,
                torch_version=torch_version,
                dtype=dtype,
                k=int(k_fix),
                limit=top_n,
                order="DESC",
            )
            if show_slow:
                slow = fast_shapes(
                    db_path,
                    hardware=hardware,
                    torch_version=torch_version,
                    dtype=dtype,
                    k=int(k_fix),
                    limit=top_n,
                    order="ASC",
                )

        def plot_spec(points: list[dict[str, Any]], title: str) -> dict[str, Any]:
            x = [p["m"] for p in points]
            y = [p["n"] for p in points]
            t = [p["tflops"] for p in points]
            return {
                "data": [
                    {
                        "type": "scattergl",
                        "mode": "markers",
                        "x": x,
                        "y": y,
                        "marker": {
                            "size": 7,
                            "color": t,
                            "colorscale": "Viridis",
                            "showscale": True,
                            "opacity": 0.9,
                        },
                        "hovertemplate": "M=%{x}<br>N=%{y}<br>TFLOPS=%{marker.color:.1f}<extra></extra>",
                    }
                ],
                "layout": {
                    "title": {"text": title, "x": 0.02},
                    "paper_bgcolor": "rgba(0,0,0,0)",
                    "plot_bgcolor": "rgba(0,0,0,0)",
                    "margin": {"l": 50, "r": 20, "t": 50, "b": 45},
                    "xaxis": {"title": "M", "gridcolor": "rgba(148,163,184,0.18)"},
                    "yaxis": {"title": "N", "gridcolor": "rgba(148,163,184,0.18)"},
                    "font": {"color": "rgba(226,232,240,0.95)"},
                },
                "config": {"responsive": True, "displaylogo": False},
            }

        context = {
            "title": "Fast & Slow Shapes",
            "hardware_list": hardware_list,
            "torch_versions": torch_versions,
            "dtype_list": dtype_list,
            "hardware": hardware,
            "torch_version": torch_version,
            "dtype": dtype,
            "k_fix": int(k_fix),
            "top_n": int(top_n),
            "show_slow": bool(show_slow),
            "has_query": has_query,
            "fast": fast,
            "slow": slow,
            "fast_plotly_spec": _safe_json_dumps(plot_spec(fast, "Fastest shapes (by mean TFLOPS)")) if fast else "",
            "slow_plotly_spec": _safe_json_dumps(plot_spec(slow, "Slowest shapes (by mean TFLOPS)")) if slow else "",
        }

        if is_hx(request):
            return render(request, "partials/fast_shapes_result.html", context)
        return render(request, "fast_shapes.html", context)

    @fastapi_app.get("/scaling", response_class=HTMLResponse)
    def scaling_page(
        request: Request,
        torch_version: str | None = None,
        dtype: str | None = None,
        metric: str = "mean_tflops",
        hardware: list[str] = Query(default_factory=list),
        configs: list[str] = Query(default_factory=list),
        sweep_dim: str = "m",
        m_fix: int = 4096,
        n_fix: int = 4096,
        k_fix: int = 4096,
        log_x: bool = False,
        log_y: bool = False,
    ) -> HTMLResponse:
        ui: dict[str, Any] = fastapi_app.state.ui_config
        torch_version = ui["baselineTorchVersion"]
        torch_versions: list[str] = []
        dtype_list: list[str] = ui["allDtypes"]
        hardware_list: list[str] = ui["hardwareList"]
        hardware_meta: dict[str, Any] = ui.get("hardwareMeta") or {}

        dtype = dtype if dtype is not None else (dtype_list[0] if dtype_list else "")

        dtype_order = {"bfloat16": 0, "float16": 1, "float8_e4m3fn": 2}
        config_options: list[dict[str, str]] = []
        for hw in hardware_list:
            dts = list(hardware_meta.get(hw, {}).get("dtypes") or [])
            dts_sorted = sorted(dts, key=lambda d: (dtype_order.get(d, 999), d))
            for dt in dts_sorted:
                value = f"{hw}|||{dt}"
                config_options.append({"value": value, "label": f"{hw} · {dt}"})

        allowed_values = {opt["value"] for opt in config_options}
        configs = [c for c in configs if c in allowed_values]
        if not configs and config_options:
            configs = [opt["value"] for opt in config_options[:2]]

        selected_configs: list[tuple[str, str]] = []
        for cfg in configs:
            if "|||" not in cfg:
                continue
            hw, dt = cfg.split("|||", 1)
            if hw in hardware_meta and dt in (hardware_meta.get(hw, {}).get("dtypes") or []):
                selected_configs.append((hw, dt))

        # Back-compat: if someone links with dtype+hardware but no configs.
        if not selected_configs and "dtype" in request.query_params:
            if hardware:
                selected_configs = [(hw, dtype) for hw in hardware]
            else:
                selected_configs = [(hw, dtype) for hw in (hardware_list[:2] if len(hardware_list) >= 2 else hardware_list)]

        has_query = "configs" in request.query_params or "dtype" in request.query_params
        rows: list[dict[str, Any]] = []
        if has_query and selected_configs:
            rows = scaling_curve_configs(
                db_path,
                configs=selected_configs,
                metric=metric,
                torch_version=torch_version,
                sweep_dim=sweep_dim,
                m_fix=int(m_fix),
                n_fix=int(n_fix),
                k_fix=int(k_fix),
            )

        sweep_dim_lower = sweep_dim.strip().lower()
        if sweep_dim_lower not in {"m", "n", "k"}:
            sweep_dim_lower = "m"

        def plot_spec() -> dict[str, Any]:
            traces = []
            series_labels = []
            for hw, dt in selected_configs:
                label = f"{hw} · {dt}"
                series_labels.append(label)
                xs = []
                ys = []
                for r in rows:
                    if r.get("hardware") != hw or r.get("dtype") != dt:
                        continue
                    xs.append(int(r[sweep_dim_lower]))
                    ys.append(float(r["tflops"]))
                traces.append(
                    {
                        "type": "scatter",
                        "mode": "lines+markers",
                        "name": label,
                        "x": xs,
                        "y": ys,
                    }
                )
            return {
                "data": traces,
                "layout": {
                    "title": {"text": f"Scaling curve (sweep {sweep_dim.upper()})", "x": 0.02},
                    "paper_bgcolor": "rgba(0,0,0,0)",
                    "plot_bgcolor": "rgba(0,0,0,0)",
                    "margin": {"l": 60, "r": 20, "t": 55, "b": 55},
                    "xaxis": {
                        "title": sweep_dim.upper(),
                        "type": "log" if log_x else "linear",
                        "gridcolor": "rgba(148,163,184,0.18)",
                    },
                    "yaxis": {
                        "title": "TFLOPS",
                        "type": "log" if log_y else "linear",
                        "gridcolor": "rgba(148,163,184,0.18)",
                    },
                    "font": {"color": "rgba(226,232,240,0.95)"},
                    "legend": {"orientation": "h", "y": -0.25},
                },
                "config": {"responsive": True, "displaylogo": False},
            }

        download_url_obj = request.url.replace(path="/download/scaling.csv")
        if torch_version:
            download_url_obj = download_url_obj.include_query_params(torch_version=torch_version)
        download_url = str(download_url_obj)

        context = {
            "title": "Scaling Curves",
            "torch_versions": torch_versions,
            "dtype_list": dtype_list,
            "hardware_list": hardware_list,
            "config_options": config_options,
            "torch_version": torch_version,
            "dtype": dtype,
            "metric": metric,
            "hardware": hardware,
            "configs": configs,
            "series_labels": [f"{hw} · {dt}" for hw, dt in selected_configs],
            "sweep_dim": sweep_dim.upper(),
            "m_fix": int(m_fix),
            "n_fix": int(n_fix),
            "k_fix": int(k_fix),
            "log_x": bool(log_x),
            "log_y": bool(log_y),
            "has_query": has_query,
            "rows": rows,
            "plotly_spec": _safe_json_dumps(plot_spec()) if rows else "",
            "download_url": download_url,
        }

        if is_hx(request):
            return render(request, "partials/scaling_result.html", context)
        return render(request, "scaling.html", context)

    @fastapi_app.get("/download/scaling.csv")
    def download_scaling_csv(
        request: Request,
        dtype: str | None = None,
        metric: str = "mean_tflops",
        torch_version: str | None = None,
        hardware: list[str] = Query(default_factory=list),
        configs: list[str] = Query(default_factory=list),
        sweep_dim: str = "m",
        m_fix: int = 4096,
        n_fix: int = 4096,
        k_fix: int = 4096,
    ) -> StreamingResponse:
        ui: dict[str, Any] = fastapi_app.state.ui_config
        hardware_meta: dict[str, Any] = ui.get("hardwareMeta") or {}
        allowed_configs: set[str] = set()
        for hw, meta in hardware_meta.items():
            for dt in meta.get("dtypes") or []:
                allowed_configs.add(f"{hw}|||{dt}")

        selected_configs: list[tuple[str, str]] = []
        for cfg in configs:
            if cfg not in allowed_configs or "|||" not in cfg:
                continue
            hw, dt = cfg.split("|||", 1)
            selected_configs.append((hw, dt))

        if not selected_configs:
            torch_versions = sorted(list(distinct_values(db_path, "torch_version")), key=_version_key)
            if torch_versions and torch_version is None:
                torch_version = torch_versions[-1]
            if dtype is None:
                raise ValueError("dtype is required when configs are not provided")
            rows = scaling_curve(
                db_path,
                dtype=dtype,
                metric=metric,
                torch_version=torch_version,
                hardware=hardware,
                sweep_dim=sweep_dim,
                m_fix=int(m_fix),
                n_fix=int(n_fix),
                k_fix=int(k_fix),
            )
        else:
            torch_version = ui.get("baselineTorchVersion")
            rows = scaling_curve_configs(
                db_path,
                configs=selected_configs,
                metric=metric,
                torch_version=torch_version,
                sweep_dim=sweep_dim,
                m_fix=int(m_fix),
                n_fix=int(n_fix),
                k_fix=int(k_fix),
            )

        def generate() -> bytes:
            buf = io.StringIO()
            writer = csv.writer(buf)
            if selected_configs:
                writer.writerow(["hardware", "dtype", "m", "n", "k", "tflops"])
            else:
                writer.writerow(["hardware", "m", "n", "k", "tflops"])
            for r in rows:
                if selected_configs:
                    writer.writerow([r["hardware"], r["dtype"], r["m"], r["n"], r["k"], f"{r['tflops']:.6g}"])
                else:
                    writer.writerow([r["hardware"], r["m"], r["n"], r["k"], f"{r['tflops']:.6g}"])
            return buf.getvalue().encode("utf-8")

        dtype_tag = dtype if dtype is not None else "multi"
        filename = f"scaling_{dtype_tag}_{metric}_sweep{sweep_dim}_M{m_fix}_N{n_fix}_K{k_fix}.csv"
        return StreamingResponse(
            iter([generate()]),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @fastapi_app.get("/compare", response_class=HTMLResponse)
    def compare_page(
        request: Request,
        torch_version: str | None = None,
        dtype: str | None = None,
        m: int = 4096,
        n: int = 4096,
        k: int = 4096,
    ) -> HTMLResponse:
        ui: dict[str, Any] = fastapi_app.state.ui_config
        torch_version = ui["baselineTorchVersion"]
        torch_versions: list[str] = []
        dtype_list: list[str] = ui["allDtypes"]
        hardware_list: list[str] = ui["hardwareList"]

        dtype = dtype if dtype is not None else (dtype_list[0] if dtype_list else "")
        has_query = "dtype" in request.query_params
        rows: list[dict[str, Any]] = []
        if has_query and dtype:
            rows = compare_hardware(db_path, dtype=dtype, torch_version=torch_version, m=int(m), n=int(n), k=int(k))

        def plot_spec() -> dict[str, Any]:
            xs = [r["hardware"] for r in rows]
            ys = [r["mean_tflops"] for r in rows]
            texts = [f"{y:.1f}" for y in ys]
            return {
                "data": [
                    {
                        "type": "bar",
                        "x": xs,
                        "y": ys,
                        "text": texts,
                        "textposition": "outside",
                        "marker": {"color": ys, "colorscale": "Viridis"},
                        "hovertemplate": "%{x}<br>Mean TFLOPS=%{y:.2f}<extra></extra>",
                    }
                ],
                "layout": {
                    "title": {"text": "Mean TFLOPS by GPU", "x": 0.02},
                    "paper_bgcolor": "rgba(0,0,0,0)",
                    "plot_bgcolor": "rgba(0,0,0,0)",
                    "margin": {"l": 60, "r": 20, "t": 55, "b": 60},
                    "yaxis": {"title": "Mean TFLOPS", "gridcolor": "rgba(148,163,184,0.18)"},
                    "font": {"color": "rgba(226,232,240,0.95)"},
                },
                "config": {"responsive": True, "displaylogo": False},
            }

        context = {
            "title": "Compare Hardware",
            "torch_versions": torch_versions,
            "dtype_list": dtype_list,
            "hardware_list": hardware_list,
            "torch_version": torch_version,
            "dtype": dtype,
            "m": int(m),
            "n": int(n),
            "k": int(k),
            "has_query": has_query,
            "rows": rows,
            "plotly_spec": _safe_json_dumps(plot_spec()) if rows else "",
        }

        if is_hx(request):
            return render(request, "partials/compare_result.html", context)
        return render(request, "compare.html", context)

    @fastapi_app.get("/compare-dtypes", response_class=HTMLResponse)
    def compare_dtypes_page(
        request: Request,
        hardware: str | None = None,
        torch_version: str | None = None,
        metric: str = "mean_tflops",
        m: int = 4096,
        n: int = 4096,
        k: int = 4096,
    ) -> HTMLResponse:
        ui: dict[str, Any] = fastapi_app.state.ui_config
        hardware_list: list[str] = ui["hardwareList"]
        hardware_meta: dict[str, Any] = ui["hardwareMeta"]
        baseline_tv: str | None = ui["baselineTorchVersion"]

        hardware = hardware if hardware is not None else (hardware_list[0] if hardware_list else "")
        if hardware not in hardware_list:
            hardware = hardware_list[0] if hardware_list else ""

        torch_versions = list(
            hardware_meta.get(hardware, {}).get("torch_versions") or ([baseline_tv] if baseline_tv else [])
        )

        if baseline_tv is None:
            torch_version = None
        else:
            if torch_version not in torch_versions:
                if baseline_tv in torch_versions:
                    torch_version = baseline_tv
                elif torch_versions:
                    torch_version = torch_versions[-1]
                else:
                    torch_version = baseline_tv

        has_query = "hardware" in request.query_params

        rows: list[dict[str, Any]] = []
        if has_query and hardware:
            rows = compare_dtypes(
                db_path,
                hardware=hardware,
                torch_version=torch_version,
                m=int(m),
                n=int(n),
                k=int(k),
            )

        metric = metric if metric in {"mean_tflops", "median_tflops", "max_tflops"} else "mean_tflops"

        def plot_spec() -> dict[str, Any]:
            xs = [r["dtype"] for r in rows]
            ys = [r[metric] for r in rows]
            texts = [f"{y:.1f}" for y in ys]
            return {
                "data": [
                    {
                        "type": "bar",
                        "x": xs,
                        "y": ys,
                        "text": texts,
                        "textposition": "outside",
                        "marker": {"color": ys, "colorscale": "Viridis"},
                        "hovertemplate": "%{x}<br>TFLOPS=%{y:.2f}<extra></extra>",
                    }
                ],
                "layout": {
                    "title": {"text": f"{metric} by dtype", "x": 0.02},
                    "paper_bgcolor": "rgba(0,0,0,0)",
                    "plot_bgcolor": "rgba(0,0,0,0)",
                    "margin": {"l": 60, "r": 20, "t": 55, "b": 60},
                    "yaxis": {"title": "TFLOPS", "gridcolor": "rgba(148,163,184,0.18)"},
                    "font": {"color": "rgba(226,232,240,0.95)"},
                },
                "config": {"responsive": True, "displaylogo": False},
            }

        context = {
            "title": "Compare Dtypes",
            "hardware_list": hardware_list,
            "torch_versions": torch_versions,
            "hardware": hardware,
            "torch_version": torch_version,
            "metric": metric,
            "m": int(m),
            "n": int(n),
            "k": int(k),
            "has_query": has_query,
            "rows": rows,
            "plotly_spec": _safe_json_dumps(plot_spec()) if rows else "",
        }

        if is_hx(request):
            return render(request, "partials/compare_dtypes_result.html", context)
        return render(request, "compare_dtypes.html", context)

    @fastapi_app.get("/compare-torch", response_class=HTMLResponse)
    def compare_torch_page(
        request: Request,
        hardware: str | None = None,
        dtype: str | None = None,
        metric: str = "mean_tflops",
        m: int = 4096,
        n: int = 4096,
        k: int = 4096,
    ) -> HTMLResponse:
        ui: dict[str, Any] = fastapi_app.state.ui_config
        full_hardware_list: list[str] = ui["hardwareList"]
        hardware_meta: dict[str, Any] = ui["hardwareMeta"]
        all_dtypes: list[str] = ui["allDtypes"]

        hardware_list: list[str] = ui["multiTorchHardware"] or full_hardware_list

        hardware = hardware if hardware is not None else (hardware_list[0] if hardware_list else "")
        if hardware not in hardware_list:
            hardware = hardware_list[0] if hardware_list else ""

        dtype_list = list(hardware_meta.get(hardware, {}).get("dtypes") or all_dtypes)
        dtype = dtype if dtype is not None and dtype in dtype_list else (dtype_list[0] if dtype_list else "")

        has_query = "hardware" in request.query_params and "dtype" in request.query_params
        rows: list[dict[str, Any]] = []
        if has_query and hardware and dtype:
            rows = compare_torch_versions(
                db_path,
                hardware=hardware,
                dtype=dtype,
                m=int(m),
                n=int(n),
                k=int(k),
            )
            rows = sorted(rows, key=lambda r: _version_key(r["torch_version"]))

        metric = metric if metric in {"mean_tflops", "median_tflops", "max_tflops"} else "mean_tflops"

        def plot_spec() -> dict[str, Any]:
            xs = [r["torch_version"] for r in rows]
            ys = [r[metric] for r in rows]
            texts = [f"{y:.1f}" for y in ys]
            return {
                "data": [
                    {
                        "type": "bar",
                        "x": xs,
                        "y": ys,
                        "text": texts,
                        "textposition": "outside",
                        "marker": {"color": ys, "colorscale": "Viridis"},
                        "hovertemplate": "%{x}<br>TFLOPS=%{y:.2f}<extra></extra>",
                    }
                ],
                "layout": {
                    "title": {"text": f"{metric} by torch version", "x": 0.02},
                    "paper_bgcolor": "rgba(0,0,0,0)",
                    "plot_bgcolor": "rgba(0,0,0,0)",
                    "margin": {"l": 60, "r": 20, "t": 55, "b": 60},
                    "yaxis": {"title": "TFLOPS", "gridcolor": "rgba(148,163,184,0.18)"},
                    "font": {"color": "rgba(226,232,240,0.95)"},
                },
                "config": {"responsive": True, "displaylogo": False},
            }

        context = {
            "title": "Compare Torch Versions",
            "hardware_list": hardware_list,
            "dtype_list": dtype_list,
            "hardware": hardware,
            "dtype": dtype,
            "metric": metric,
            "m": int(m),
            "n": int(n),
            "k": int(k),
            "has_query": has_query,
            "rows": rows,
            "plotly_spec": _safe_json_dumps(plot_spec()) if rows else "",
        }

        if is_hx(request):
            return render(request, "partials/compare_torch_result.html", context)
        return render(request, "compare_torch.html", context)

    @fastapi_app.get("/compare-configs", response_class=HTMLResponse)
    def compare_configs_page(
        request: Request,
        hardware_a: str | None = None,
        torch_version_a: str | None = None,
        dtype_a: str | None = None,
        hardware_b: str | None = None,
        torch_version_b: str | None = None,
        dtype_b: str | None = None,
        metric: str = "mean_tflops",
        k: int = 4096,
        bins: int = 50,
        top_n: int = 30,
        min_tflops: float = 1.0,
    ) -> HTMLResponse:
        ui: dict[str, Any] = fastapi_app.state.ui_config
        hardware_list: list[str] = ui["hardwareList"]
        hardware_meta: dict[str, Any] = ui["hardwareMeta"]
        all_dtypes: list[str] = ui["allDtypes"]
        baseline_tv: str | None = ui["baselineTorchVersion"]

        preferred_a = "NVIDIA H100 80GB HBM3"
        preferred_b = "NVIDIA H200"

        default_a = preferred_a if preferred_a in hardware_list else (hardware_list[0] if hardware_list else "")
        if hardware_a not in hardware_list:
            hardware_a = default_a

        default_b = ""
        if hardware_list:
            if preferred_b in hardware_list and preferred_b != hardware_a:
                default_b = preferred_b
            else:
                default_b = next((hw for hw in hardware_list if hw != hardware_a), hardware_a or hardware_list[0])
        if hardware_b not in hardware_list:
            hardware_b = default_b

        dtype_list_a = list(hardware_meta.get(hardware_a, {}).get("dtypes") or all_dtypes)
        dtype_list_b = list(hardware_meta.get(hardware_b, {}).get("dtypes") or all_dtypes)
        dtype_a = dtype_a if dtype_a is not None and dtype_a in dtype_list_a else (dtype_list_a[0] if dtype_list_a else "")
        dtype_b = dtype_b if dtype_b is not None and dtype_b in dtype_list_b else (dtype_list_b[0] if dtype_list_b else "")

        torch_versions_a = list(
            hardware_meta.get(hardware_a, {}).get("torch_versions") or ([baseline_tv] if baseline_tv else [])
        )
        torch_versions_b = list(
            hardware_meta.get(hardware_b, {}).get("torch_versions") or ([baseline_tv] if baseline_tv else [])
        )

        def _pick_torch(
            torch_version: str | None, torch_versions: list[str], baseline: str | None
        ) -> str | None:
            if baseline is None:
                return None
            if torch_version in torch_versions:
                return torch_version
            if baseline in torch_versions:
                return baseline
            if torch_versions:
                return torch_versions[-1]
            return baseline

        torch_version_a = _pick_torch(torch_version_a, torch_versions_a, baseline_tv)
        torch_version_b = _pick_torch(torch_version_b, torch_versions_b, baseline_tv)

        metric = metric if metric in {"mean_tflops", "median_tflops", "max_tflops"} else "mean_tflops"
        bins_val = max(20, min(int(bins), 120))
        top_n_val = max(10, min(int(top_n), 200))
        k_val = int(k) if int(k) else None
        min_tflops_val = max(0.0, float(min_tflops))

        has_query = (
            "hardware_a" in request.query_params
            and "dtype_a" in request.query_params
            and "hardware_b" in request.query_params
            and "dtype_b" in request.query_params
        )

        analysis: dict[str, Any] | None = None
        plotly_spec = ""
        if has_query and hardware_a and dtype_a and hardware_b and dtype_b:
            analysis = compare_configs_speedup(
                db_path,
                hardware_a=hardware_a,
                dtype_a=dtype_a,
                torch_version_a=torch_version_a,
                hardware_b=hardware_b,
                dtype_b=dtype_b,
                torch_version_b=torch_version_b,
                metric=metric,
                k=k_val,
                histogram_bins=bins_val,
                histogram_min=0.5,
                histogram_max=3.0,
                extremes_limit=top_n_val,
                min_tflops=min_tflops_val,
            )

            hist = analysis["histogram"]
            bins_used = int(hist["bins"])
            min_v = float(hist["min"])
            max_v = float(hist["max"])
            span = max(1e-9, max_v - min_v)
            width = span / bins_used
            counts = {int(d["bucket"]): int(d["count"]) for d in hist["data"]}
            xs = [min_v + (i + 0.5) * width for i in range(bins_used)]
            ys = [counts.get(i, 0) for i in range(bins_used)]

            plot = {
                "data": [
                    {
                        "type": "bar",
                        "x": xs,
                        "y": ys,
                        "hovertemplate": "speedup≈%{x:.2f}x<br>count=%{y}<extra></extra>",
                        "marker": {"color": "rgba(56,189,248,0.75)"},
                    }
                ],
                "layout": {
                    "title": {"text": "Speedup histogram (A / B, clipped to 0.5–3.0×)", "x": 0.02},
                    "paper_bgcolor": "rgba(0,0,0,0)",
                    "plot_bgcolor": "rgba(0,0,0,0)",
                    "margin": {"l": 60, "r": 20, "t": 55, "b": 55},
                    "xaxis": {"title": "speedup (×)", "gridcolor": "rgba(148,163,184,0.18)"},
                    "yaxis": {"title": "count", "gridcolor": "rgba(148,163,184,0.18)"},
                    "font": {"color": "rgba(226,232,240,0.95)"},
                },
                "config": {"responsive": True, "displaylogo": False},
            }
            plotly_spec = _safe_json_dumps(plot)

        context = {
            "title": "Compare Configs",
            "hardware_list": hardware_list,
            "hardware_a": hardware_a,
            "hardware_b": hardware_b,
            "dtype_list_a": dtype_list_a,
            "dtype_list_b": dtype_list_b,
            "torch_versions_a": torch_versions_a,
            "torch_versions_b": torch_versions_b,
            "torch_version_a": torch_version_a,
            "torch_version_b": torch_version_b,
            "dtype_a": dtype_a,
            "dtype_b": dtype_b,
            "metric": metric,
            "k": int(k),
            "bins": int(bins_val),
            "top_n": int(top_n_val),
            "min_tflops": float(min_tflops_val),
            "has_query": has_query,
            "analysis": analysis,
            "plotly_spec": plotly_spec,
        }

        if is_hx(request):
            return render(request, "partials/compare_configs_result.html", context)
        return render(request, "compare_configs.html", context)

    @fastapi_app.get("/leaderboard", response_class=HTMLResponse)
    def leaderboard_page(
        request: Request,
        hardware: str | None = None,
        torch_version: str | None = None,
        dtype: str | None = None,
        metric: str = "mean_tflops",
        limit: int = 500,
    ) -> HTMLResponse:
        ui: dict[str, Any] = fastapi_app.state.ui_config
        hardware_list: list[str] = ui["hardwareList"]
        hardware_meta: dict[str, Any] = ui["hardwareMeta"]
        all_dtypes: list[str] = ui["allDtypes"]
        baseline_tv: str | None = ui["baselineTorchVersion"]

        hardware = hardware if hardware is not None else (hardware_list[0] if hardware_list else "")
        if hardware not in hardware_list:
            hardware = hardware_list[0] if hardware_list else ""

        dtype_list = list(hardware_meta.get(hardware, {}).get("dtypes") or all_dtypes)
        torch_versions = list(
            hardware_meta.get(hardware, {}).get("torch_versions") or ([baseline_tv] if baseline_tv else [])
        )

        dtype = dtype if dtype is not None and dtype in dtype_list else (dtype_list[0] if dtype_list else "")

        if baseline_tv is None:
            torch_version = None
        else:
            if torch_version not in torch_versions:
                if baseline_tv in torch_versions:
                    torch_version = baseline_tv
                elif torch_versions:
                    torch_version = torch_versions[-1]
                else:
                    torch_version = baseline_tv

        metric = metric if metric in {"mean_tflops", "median_tflops", "max_tflops"} else "mean_tflops"
        limit_val = max(10, min(int(limit), 5000))

        has_query = "hardware" in request.query_params and "dtype" in request.query_params
        rows: list[dict[str, Any]] = []
        plotly_spec = ""
        if has_query and hardware and dtype:
            rows = top_shapes(
                db_path,
                hardware=hardware,
                dtype=dtype,
                metric=metric,
                torch_version=torch_version,
                limit=limit_val,
            )

            xs = [2 * r["m"] * r["n"] * r["k"] for r in rows]
            ys = [r[metric] for r in rows]
            ks = [r["k"] for r in rows]
            labels = [f"{r['m']}×{r['n']}×{r['k']}" for r in rows]
            plot = {
                "data": [
                    {
                        "type": "scattergl",
                        "mode": "markers",
                        "x": xs,
                        "y": ys,
                        "text": labels,
                        "marker": {
                            "size": 7,
                            "color": ks,
                            "colorscale": "Viridis",
                            "showscale": True,
                            "opacity": 0.9,
                            "colorbar": {"title": "K"},
                        },
                        "hovertemplate": "FLOPs=%{x:.3g}<br>TFLOPS=%{y:.2f}<br>%{text}<extra></extra>",
                    }
                ],
                "layout": {
                    "title": {"text": f"Top shapes by {metric}", "x": 0.02},
                    "paper_bgcolor": "rgba(0,0,0,0)",
                    "plot_bgcolor": "rgba(0,0,0,0)",
                    "margin": {"l": 60, "r": 20, "t": 55, "b": 55},
                    "xaxis": {
                        "title": "FLOPs (2·M·N·K)",
                        "type": "log",
                        "gridcolor": "rgba(148,163,184,0.18)",
                    },
                    "yaxis": {"title": "TFLOPS", "gridcolor": "rgba(148,163,184,0.18)"},
                    "font": {"color": "rgba(226,232,240,0.95)"},
                },
                "config": {"responsive": True, "displaylogo": False},
            }
            plotly_spec = _safe_json_dumps(plot)

        context = {
            "title": "Leaderboard",
            "hardware_list": hardware_list,
            "torch_versions": torch_versions,
            "dtype_list": dtype_list,
            "hardware": hardware,
            "torch_version": torch_version,
            "dtype": dtype,
            "metric": metric,
            "limit": int(limit_val),
            "has_query": has_query,
            "rows": rows,
            "plotly_spec": plotly_spec,
        }

        if is_hx(request):
            return render(request, "partials/leaderboard_result.html", context)
        return render(request, "leaderboard.html", context)

    @fastapi_app.get("/speedup", response_class=HTMLResponse)
    def speedup_page(
        request: Request,
        hardware: str | None = None,
        torch_version: str | None = None,
        metric: str = "mean_tflops",
        k: int = 4096,
        bins: int = 50,
        top_n: int = 30,
    ) -> HTMLResponse:
        ui: dict[str, Any] = fastapi_app.state.ui_config
        full_hardware_list: list[str] = ui["hardwareList"]
        hardware_meta: dict[str, Any] = ui["hardwareMeta"]
        baseline_tv: str | None = ui["baselineTorchVersion"]

        speedup_hardware = [
            hw
            for hw in full_hardware_list
            if {
                "bfloat16",
                "float8_e4m3fn",
            }.issubset(set(hardware_meta.get(hw, {}).get("dtypes") or []))
        ]

        hardware = hardware if hardware is not None else (speedup_hardware[0] if speedup_hardware else "")
        if hardware not in speedup_hardware:
            hardware = speedup_hardware[0] if speedup_hardware else ""

        torch_versions = list(
            hardware_meta.get(hardware, {}).get("torch_versions") or ([baseline_tv] if baseline_tv else [])
        )
        if baseline_tv is None:
            torch_version = None
        else:
            if torch_version not in torch_versions:
                if baseline_tv in torch_versions:
                    torch_version = baseline_tv
                elif torch_versions:
                    torch_version = torch_versions[-1]
                else:
                    torch_version = baseline_tv

        metric = metric if metric in {"mean_tflops", "median_tflops", "max_tflops"} else "mean_tflops"
        bins_val = max(20, min(int(bins), 120))
        top_n_val = max(10, min(int(top_n), 200))
        k_val = int(k) if int(k) else None

        has_query = "hardware" in request.query_params
        analysis: dict[str, Any] | None = None
        plotly_spec = ""
        if has_query and hardware:
            analysis = fp8_vs_bf16_speedup(
                db_path,
                hardware=hardware,
                torch_version=torch_version,
                metric=metric,
                k=k_val,
                histogram_bins=bins_val,
                histogram_min=0.5,
                histogram_max=3.0,
                extremes_limit=top_n_val,
            )
            hist = analysis["histogram"]
            bins_used = int(hist["bins"])
            min_v = float(hist["min"])
            max_v = float(hist["max"])
            span = max(1e-9, max_v - min_v)
            width = span / bins_used
            counts = {int(d["bucket"]): int(d["count"]) for d in hist["data"]}
            xs = [min_v + (i + 0.5) * width for i in range(bins_used)]
            ys = [counts.get(i, 0) for i in range(bins_used)]

            plot = {
                "data": [
                    {
                        "type": "bar",
                        "x": xs,
                        "y": ys,
                        "hovertemplate": "speedup≈%{x:.2f}x<br>count=%{y}<extra></extra>",
                        "marker": {"color": "rgba(56,189,248,0.75)"},
                    }
                ],
                "layout": {
                    "title": {"text": "FP8 / BF16 speedup histogram (clipped to 0.5–3.0×)", "x": 0.02},
                    "paper_bgcolor": "rgba(0,0,0,0)",
                    "plot_bgcolor": "rgba(0,0,0,0)",
                    "margin": {"l": 60, "r": 20, "t": 55, "b": 55},
                    "xaxis": {"title": "speedup (×)", "gridcolor": "rgba(148,163,184,0.18)"},
                    "yaxis": {"title": "count", "gridcolor": "rgba(148,163,184,0.18)"},
                    "font": {"color": "rgba(226,232,240,0.95)"},
                },
                "config": {"responsive": True, "displaylogo": False},
            }
            plotly_spec = _safe_json_dumps(plot)

        context = {
            "title": "FP8 vs BF16 Speedup",
            "hardware_list": speedup_hardware,
            "torch_versions": torch_versions,
            "hardware": hardware,
            "torch_version": torch_version,
            "metric": metric,
            "k": int(k),
            "bins": int(bins_val),
            "top_n": int(top_n_val),
            "has_query": has_query,
            "analysis": analysis,
            "plotly_spec": plotly_spec,
        }

        if is_hx(request):
            return render(request, "partials/speedup_result.html", context)
        return render(request, "speedup.html", context)

    @fastapi_app.get("/stability", response_class=HTMLResponse)
    def stability_page(
        request: Request,
        hardware: str | None = None,
        torch_version: str | None = None,
        dtype: str | None = None,
        sort_by: str = "max_over_median",
        min_denominator: float = 1.0,
        limit: int = 200,
    ) -> HTMLResponse:
        ui: dict[str, Any] = fastapi_app.state.ui_config
        hardware_list: list[str] = ui["hardwareList"]
        hardware_meta: dict[str, Any] = ui["hardwareMeta"]
        all_dtypes: list[str] = ui["allDtypes"]
        baseline_tv: str | None = ui["baselineTorchVersion"]

        hardware = hardware if hardware is not None else (hardware_list[0] if hardware_list else "")
        if hardware not in hardware_list:
            hardware = hardware_list[0] if hardware_list else ""

        dtype_list = list(hardware_meta.get(hardware, {}).get("dtypes") or all_dtypes)
        torch_versions = list(
            hardware_meta.get(hardware, {}).get("torch_versions") or ([baseline_tv] if baseline_tv else [])
        )

        dtype = dtype if dtype is not None and dtype in dtype_list else (dtype_list[0] if dtype_list else "")

        if baseline_tv is None:
            torch_version = None
        else:
            if torch_version not in torch_versions:
                if baseline_tv in torch_versions:
                    torch_version = baseline_tv
                elif torch_versions:
                    torch_version = torch_versions[-1]
                else:
                    torch_version = baseline_tv

        sort_by = sort_by if sort_by in {"max_over_mean", "max_over_median", "max_minus_median"} else "max_over_median"
        limit_val = max(10, min(int(limit), 5000))
        min_den_val = max(0.0, float(min_denominator))

        has_query = "hardware" in request.query_params and "dtype" in request.query_params
        rows: list[dict[str, Any]] = []
        if has_query and hardware and dtype:
            rows = stability_outliers(
                db_path,
                hardware=hardware,
                dtype=dtype,
                sort_by=sort_by,
                torch_version=torch_version,
                min_denominator=min_den_val,
                limit=limit_val,
            )

        context = {
            "title": "Stability",
            "hardware_list": hardware_list,
            "torch_versions": torch_versions,
            "dtype_list": dtype_list,
            "hardware": hardware,
            "torch_version": torch_version,
            "dtype": dtype,
            "sort_by": sort_by,
            "min_denominator": float(min_den_val),
            "limit": int(limit_val),
            "has_query": has_query,
            "rows": rows,
        }

        if is_hx(request):
            return render(request, "partials/stability_result.html", context)
        return render(request, "stability.html", context)

    @fastapi_app.get("/browse", response_class=HTMLResponse)
    def browse_page(
        request: Request,
        dtype_values: list[str] = Query(default_factory=list),
        torch_versions: list[str] = Query(default_factory=list),
        hardware_values: list[str] = Query(default_factory=list),
        m: int = 0,
        n: int = 0,
        k: int = 0,
        order_by: str = "mean_desc",
        limit: int = 50_000,
    ) -> HTMLResponse:
        ui: dict[str, Any] = fastapi_app.state.ui_config
        dtype_list: list[str] = ui["allDtypes"]
        hardware_list: list[str] = ui["hardwareList"]
        all_torch_versions: list[str] = ui["allTorchVersions"]
        baseline_tv: str | None = ui["baselineTorchVersion"]
        multi_torch_hw = set(ui["multiTorchHardware"])

        if not dtype_values:
            dtype_values = dtype_list[:]
        if not hardware_values:
            hardware_values = hardware_list[:]

        allow_multi_torch = len(hardware_values) == 1 and hardware_values[0] in multi_torch_hw
        if allow_multi_torch:
            torch_version_list = all_torch_versions[:]
            if torch_versions:
                torch_versions = [tv for tv in torch_versions if tv in torch_version_list]
            if torch_version_list and not torch_versions:
                torch_versions = [torch_version_list[-1]]
        else:
            torch_version_list = [baseline_tv] if baseline_tv else (all_torch_versions[-1:] if all_torch_versions else [])
            torch_versions = torch_version_list[:1] if torch_version_list else []

        has_query = bool(request.query_params)
        m_val = int(m) if int(m) else None
        n_val = int(n) if int(n) else None
        k_val = int(k) if int(k) else None
        limit_val = None if int(limit) == 0 else max(1, min(int(limit), 500_000))

        rows: list[dict[str, Any]] = []
        if has_query:
            rows = browse_rows(
                db_path,
                dtype_values=dtype_values,
                torch_versions=torch_versions,
                hardware_values=hardware_values,
                m=m_val,
                n=n_val,
                k=k_val,
                order_by=order_by,
                limit=limit_val,
            )

        download_url = str(request.url.replace(path="/download/browse.csv"))

        context = {
            "title": "Browse & Export",
            "dtype_list": dtype_list,
            "torch_version_list": torch_version_list,
            "hardware_list": hardware_list,
            "dtype_values": dtype_values,
            "torch_versions": torch_versions,
            "hardware_values": hardware_values,
            "m": int(m),
            "n": int(n),
            "k": int(k),
            "order_by": order_by,
            "limit": int(limit),
            "has_query": has_query,
            "rows": rows,
            "download_url": download_url,
        }

        if is_hx(request):
            return render(request, "partials/browse_result.html", context)
        return render(request, "browse.html", context)

    @fastapi_app.get("/download/browse.csv")
    def download_browse_csv(
        request: Request,
        dtype_values: list[str] = Query(default_factory=list),
        torch_versions: list[str] = Query(default_factory=list),
        hardware_values: list[str] = Query(default_factory=list),
        m: int = 0,
        n: int = 0,
        k: int = 0,
        order_by: str = "mean_desc",
        limit: int = 50_000,
    ) -> StreamingResponse:
        dtype_list = list(distinct_values(db_path, "dtype"))
        torch_version_list = sorted(list(distinct_values(db_path, "torch_version")), key=_version_key)
        python_version_list = list(distinct_values(db_path, "python_version"))
        hardware_list = list(distinct_values(db_path, "hardware"))
        if not dtype_values:
            dtype_values = dtype_list[:]
        if torch_version_list and not torch_versions:
            torch_versions = [torch_version_list[-1]]
        if not hardware_values:
            hardware_values = hardware_list[:]

        m_val = int(m) if int(m) else None
        n_val = int(n) if int(n) else None
        k_val = int(k) if int(k) else None
        limit_val = None if int(limit) == 0 else max(1, min(int(limit), 500_000))

        rows = browse_rows(
            db_path,
            dtype_values=dtype_values,
            torch_versions=torch_versions,
            hardware_values=hardware_values,
            m=m_val,
            n=n_val,
            k=k_val,
            order_by=order_by,
            limit=limit_val,
        )

        def generate() -> bytes:
            buf = io.StringIO()
            writer = csv.writer(buf)
            header = []
            if python_version_list:
                header.append("python_version")
            if torch_version_list:
                header.append("torch_version")
            header.extend(["hardware", "dtype", "m", "n", "k", "mean_tflops", "median_tflops", "max_tflops"])
            writer.writerow(header)
            for r in rows:
                row_out: list[object] = []
                if python_version_list:
                    row_out.append(r.get("python_version") or "")
                if torch_version_list:
                    row_out.append(r.get("torch_version") or "")
                row_out.extend(
                    [
                        r["hardware"],
                        r["dtype"],
                        r["m"],
                        r["n"],
                        r["k"],
                        f"{r['mean_tflops']:.6g}",
                        f"{r['median_tflops']:.6g}",
                        f"{r['max_tflops']:.6g}",
                    ]
                )
                writer.writerow(
                    row_out
                )
            return buf.getvalue().encode("utf-8")

        filename = "matmul_results_filtered.csv"
        return StreamingResponse(
            iter([generate()]),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return fastapi_app


app = create_app()
