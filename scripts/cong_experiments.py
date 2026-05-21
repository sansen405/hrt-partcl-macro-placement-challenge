"""Congestion surrogate ablation for TierPlace.

Tests 5 configurations to figure out where the 0.6+ proxy contribution
from congestion is coming from and what to change.

Configurations:

    A: baseline               — current tierplace.py, untouched.
    B: legacy=0               — Phases 1–3 use a no-op congestion loss
                                (legacy bbox surrogate is anti-correlated
                                with real TILOS congestion per the
                                _congestion_loss docstring).
    C: TILOS in P1–3          — Phases 1–3 use the TILOS-aligned
                                _congestion_loss (no macro blockage,
                                since macros aren't legal yet).
    D: longer P4              — baseline P1–3, but joint_polish_iters=500
                                (default 120). P4 is the only phase
                                running TILOS-aligned congestion on hard
                                macros.
    E: TILOS + longer P4      — combination of C and D.

Each config runs on a small/medium/large stress-tier triple
(ibm01 / ibm07 / ibm14) and prints a comparison table.

Usage:
    uv run python scripts/cong_experiments.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

# Make tierplace importable as a top-level module (same trick as
# structureplace.py uses, since submissions/will_seed isn't a package).
_ROOT = Path(__file__).resolve().parent.parent
_TIERPLACE_DIR = _ROOT / "submissions" / "will_seed"
sys.path.insert(0, str(_TIERPLACE_DIR))

import tierplace  # noqa: E402
from tierplace import AnalyticalPlacer  # noqa: E402

# Importing tierplace installs the loader patch that attaches plc to
# benchmark objects, so we can call compute_proxy_cost on the result.
from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402


BENCHMARKS = ["ibm01", "ibm07", "ibm14"]
TESTCASE_ROOT = _ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"


# ---------------------------------------------------------------------------
# Patch factories — each returns a callable to install in place of
# tierplace._legacy_bbox_congestion_loss for the duration of one run.
# ---------------------------------------------------------------------------

_orig_legacy = tierplace._legacy_bbox_congestion_loss
_tilos = tierplace._congestion_loss


def _zero_loss(pos_all, net_idx, net_mask, gx, gy, bw, bh, gamma):
    """No-op congestion loss; preserves the autograd graph (zero-valued)."""
    return pos_all.sum() * 0.0


def _tilos_no_macro(pos_all, net_idx, net_mask, gx, gy, bw, bh, gamma):
    """TILOS-aligned _congestion_loss without the macro-blockage term.

    Phases 1–3 don't have legal macro positions yet, so the macro
    blockage doesn't apply; the per-net L-shape routing term is what
    we care about and it's correctly correlated.
    """
    return _tilos(pos_all, net_idx, net_mask, gx, gy, bw, bh, gamma)


# (config_name, congestion_patch, AnalyticalPlacer kwargs override)
CONFIGS: List[Tuple[str, Callable, Dict]] = [
    ("A: baseline", _orig_legacy, {}),
    ("B: legacy=0", _zero_loss, {}),
    ("C: TILOS in P1-3", _tilos_no_macro, {}),
    ("D: longer P4 (500)", _orig_legacy, {"joint_polish_iters": 500}),
    ("E: TILOS + longer P4", _tilos_no_macro, {"joint_polish_iters": 500}),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_one(
    name: str,
    patch: Callable,
    extra_kwargs: Dict,
    benchmark_name: str,
) -> Dict:
    bench_dir = TESTCASE_ROOT / benchmark_name
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    tierplace._legacy_bbox_congestion_loss = patch
    try:
        placer = AnalyticalPlacer(
            seed=42,
            verbose=False,
            gate_with_real_proxy=True,
            **extra_kwargs,
        )
        t0 = time.time()
        placement = placer.place(benchmark)
        runtime = time.time() - t0
        costs = compute_proxy_cost(placement, benchmark, plc)
    finally:
        tierplace._legacy_bbox_congestion_loss = _orig_legacy

    return {
        "config": name,
        "benchmark": benchmark_name,
        "proxy": float(costs["proxy_cost"]),
        "wl": float(costs["wirelength_cost"]),
        "den": float(costs["density_cost"]),
        "cong": float(costs["congestion_cost"]),
        "overlaps": int(costs["overlap_count"]),
        "runtime": runtime,
    }


def main() -> None:
    print("=" * 86)
    print(f"TierPlace congestion surrogate ablation on {', '.join(BENCHMARKS)}")
    print("=" * 86)
    print()

    results: List[Dict] = []
    for cname, patch, kwargs in CONFIGS:
        for bench in BENCHMARKS:
            print(f"  {bench:6s}  {cname:<24s} ...", end=" ", flush=True)
            r = run_one(cname, patch, kwargs, bench)
            results.append(r)
            print(
                f"proxy={r['proxy']:.4f}  "
                f"(wl={r['wl']:.3f} den={r['den']:.3f} cong={r['cong']:.3f}) "
                f"overlaps={r['overlaps']}  [{r['runtime']:.1f}s]",
                flush=True,
            )

    # Per-benchmark comparison vs baseline
    print()
    print("=" * 86)
    print("Per-benchmark deltas (vs baseline A)")
    print("=" * 86)
    for bench in BENCHMARKS:
        rows = [r for r in results if r["benchmark"] == bench]
        baseline = next(r for r in rows if r["config"] == "A: baseline")
        print(f"\n  {bench}")
        print(
            f"    {'config':<24s}  {'proxy':>8s}  {'Δproxy':>8s}  "
            f"{'cong':>7s}  {'Δcong':>8s}  {'time':>6s}"
        )
        for r in rows:
            d_proxy = r["proxy"] - baseline["proxy"]
            d_cong = r["cong"] - baseline["cong"]
            print(
                f"    {r['config']:<24s}  {r['proxy']:>8.4f}  {d_proxy:>+8.4f}  "
                f"{r['cong']:>7.3f}  {d_cong:>+8.3f}  {r['runtime']:>5.1f}s"
            )

    # Average across the trio
    print()
    print("=" * 86)
    print("Average proxy across the trio")
    print("=" * 86)
    baseline_avg = sum(
        r["proxy"] for r in results if r["config"] == "A: baseline"
    ) / len(BENCHMARKS)
    for cname, _, _ in CONFIGS:
        rows = [r for r in results if r["config"] == cname]
        avg = sum(r["proxy"] for r in rows) / len(rows)
        delta = avg - baseline_avg
        print(f"  {cname:<24s}  avg={avg:.4f}  Δ={delta:+.4f}")
    print()


if __name__ == "__main__":
    main()
