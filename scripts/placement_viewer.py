#!/usr/bin/env python3
"""
Serve an EDA-style web UI for macro placement (placement_ui/).

  # Final placement from your placer
  uv run python scripts/placement_viewer.py --benchmark ibm01 --placer submissions/will_seed/tierplaceheavy.py

  # Initial placement only (no placer run)
  uv run python scripts/placement_viewer.py --benchmark ibm01

Then open http://127.0.0.1:8765/ in a browser.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import urllib.parse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
UI_DIR = REPO_ROOT / "placement_ui"


def _load_placer(path: Path):
    path = path.resolve()
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load placer from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for attr in vars(mod).values():
        if (
            isinstance(attr, type)
            and attr.__module__ == path.stem
            and callable(getattr(attr, "place", None))
        ):
            return attr()
    raise RuntimeError(f"No placer class with place() in {path}")


def build_payload(benchmark, placement: torch.Tensor, plc) -> dict:
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    die_area = max(cw * ch, 1e-12)
    nm = benchmark.num_macros
    nh = benchmark.num_hard_macros
    macros = []
    pos = placement.detach().float().cpu()
    sizes = benchmark.macro_sizes.detach().float().cpu()
    fixed = benchmark.macro_fixed.detach().cpu()
    names = benchmark.macro_names
    for i in range(nm):
        w, h = float(sizes[i, 0]), float(sizes[i, 1])
        x, y = float(pos[i, 0]), float(pos[i, 1])
        area = w * h
        macros.append(
            {
                "id": i,
                "name": names[i] if i < len(names) else f"M{i}",
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "fixed": bool(fixed[i].item()),
                "soft": i >= nh,
                "die_footprint_pct": 100.0 * area / die_area,
            }
        )
    ports = []
    pp = benchmark.port_positions
    if pp is not None and pp.numel() > 0:
        pcpu = pp.detach().float().cpu()
        for j in range(pcpu.shape[0]):
            ports.append({"x": float(pcpu[j, 0]), "y": float(pcpu[j, 1])})

    meta = {"benchmark": benchmark.name}
    if plc is not None:
        try:
            from macro_place.objective import compute_proxy_cost

            costs = compute_proxy_cost(placement.detach().cpu().float(), benchmark, plc)
            meta["proxy_cost"] = float(costs["proxy_cost"])
            meta["valid"] = int(costs.get("overlap_count", 0)) == 0
        except Exception:
            meta["proxy_cost"] = None
            meta["valid"] = None

    return {
        "canvas": {"width": cw, "height": ch},
        "grid": {"rows": benchmark.grid_rows, "cols": benchmark.grid_cols},
        "macros": macros,
        "ports": ports,
        "meta": meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TierPlace floorplan web viewer")
    parser.add_argument("--benchmark", "-b", default="ibm01", help="ICCAD04 benchmark name")
    parser.add_argument(
        "--placer",
        "-p",
        default=None,
        help="Path to placer .py (omit to show initial placement from testcase)",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if not UI_DIR.is_dir():
        print(f"Missing UI directory: {UI_DIR}", file=sys.stderr)
        sys.exit(1)

    testcase_root = REPO_ROOT / "external/MacroPlacement/Testcases/ICCAD04"
    bench_dir = testcase_root / args.benchmark
    if not bench_dir.is_dir():
        print(f"Benchmark not found: {bench_dir}\nRun: git submodule update --init external/MacroPlacement", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, str(REPO_ROOT))
    from macro_place.loader import load_benchmark_from_dir

    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    if args.placer:
        placer = _load_placer(Path(args.placer))
        placement = placer.place(benchmark)
    else:
        placement = benchmark.macro_positions.clone()

    if not isinstance(placement, torch.Tensor):
        placement = torch.as_tensor(placement)

    data = build_payload(benchmark, placement, plc)
    data_bytes = json.dumps(data).encode("utf-8")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(UI_DIR), **kw)

        def log_message(self, fmt, *log_args):
            print("%s - %s" % (self.log_date_time_string(), fmt % log_args))

        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path
            if path == "/api/placement":
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data_bytes)))
                self.end_headers()
                self.wfile.write(data_bytes)
            else:
                super().do_GET()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving TierPlace UI at http://{args.host}:{args.port}/")
    print(f"  benchmark={args.benchmark}  placer={'initial' if not args.placer else args.placer}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
