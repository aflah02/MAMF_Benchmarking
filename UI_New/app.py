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

from mamf_db import (
    browse_rows,
    compare_dtypes,
    compare_hardware,
    compare_torch_versions,
    db_stats,
    distinct_values,
    distinct_values_for_hardware,
    fast_shapes,
    hardware_coverage,
    lookup_shape,
    scaling_curve,
)

APP_TITLE = "MAMF Explorer (New UI)"

ALL_HARDWARE_VALUE = "__all__"
ALL_HARDWARE_LABEL = "All GPUs"
BASELINE_TORCH_VERSION = "2.9.0"


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

    multi_torch_hardware = [hw for hw in hardware_list if len(hardware_meta.get(hw, {}).get("torch_versions", [])) > 1]

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
            "db_path_display": Path(db_path).name,
            "ui_config_json": _safe_json_dumps(fastapi_app.state.ui_config),
        }
        return templates.TemplateResponse(template_name, {**base, **context})

    def is_hx(request: Request) -> bool:
        return request.headers.get("hx-request", "").lower() == "true"

    @fastapi_app.get("/health", response_class=PlainTextResponse)
    def health() -> str:
        return "ok"

    @fastapi_app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        stats = db_stats(db_path)
        coverage = hardware_coverage(db_path)
        return render(
            request,
            "home.html",
            {
                "title": "Home",
                "stats": stats,
                "coverage": coverage,
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

        dtype = dtype if dtype is not None else (dtype_list[0] if dtype_list else "")
        if not hardware:
            hardware = hardware_list[:2] if len(hardware_list) >= 2 else hardware_list[:1]

        has_query = "dtype" in request.query_params
        rows: list[dict[str, Any]] = []
        if has_query and dtype:
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

        sweep_dim_lower = sweep_dim.strip().lower()
        if sweep_dim_lower not in {"m", "n", "k"}:
            sweep_dim_lower = "m"

        def plot_spec() -> dict[str, Any]:
            traces = []
            for hw in hardware:
                xs = []
                ys = []
                for r in rows:
                    if r["hardware"] != hw:
                        continue
                    xs.append(int(r[sweep_dim_lower]))
                    ys.append(float(r["tflops"]))
                traces.append(
                    {
                        "type": "scatter",
                        "mode": "lines+markers",
                        "name": hw,
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
            "torch_version": torch_version,
            "dtype": dtype,
            "metric": metric,
            "hardware": hardware,
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
        dtype: str,
        metric: str = "mean_tflops",
        torch_version: str | None = None,
        hardware: list[str] = Query(default_factory=list),
        sweep_dim: str = "m",
        m_fix: int = 4096,
        n_fix: int = 4096,
        k_fix: int = 4096,
    ) -> StreamingResponse:
        torch_versions = sorted(list(distinct_values(db_path, "torch_version")), key=_version_key)
        if torch_versions and torch_version is None:
            torch_version = torch_versions[-1]
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

        def generate() -> bytes:
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["hardware", "m", "n", "k", "tflops"])
            for r in rows:
                writer.writerow([r["hardware"], r["m"], r["n"], r["k"], f"{r['tflops']:.6g}"])
            return buf.getvalue().encode("utf-8")

        filename = f"scaling_{dtype}_{metric}_sweep{sweep_dim}_M{m_fix}_N{n_fix}_K{k_fix}.csv"
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
