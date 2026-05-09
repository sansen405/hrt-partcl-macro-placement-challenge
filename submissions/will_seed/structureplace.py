"""StructurePlace — TierPlace driven by multiple initial macro structures.

TierPlace bootstraps every benchmark from a single initial layout
(``_uniform_spread`` — a uniform grid covering the canvas).  This is fine
when the wirelength + density landscape has a smooth basin near uniform,
but it leaves performance on the table whenever the real proxy minimum
sits closer to a *structurally different* layout: a ring around the
boundary (low cross-canvas wirelength), a ``+`` carved through the
middle (clean quadrant channels for soft macros), or a few compact
clusters (tight macro-to-macro nets).

This module sweeps several **initial archetypes** through the same
TierPlace 5-phase pipeline and keeps the placement with the lowest real
TILOS proxy cost.  Each archetype seeds Adam with a structurally
different starting point, which drives the gradient descent into a
different local minimum — exactly the scenario where a single uniform
seed gets stuck.

Archetypes (3 new + the existing uniform = 4):

    * ``uniform``    — Even grid spread (current TierPlace default).
    * ``perimeter``  — Concentric rings from the outer boundary inward.
                       Largest macros land on the outer ring; the centre
                       is left clear for soft cells.
    * ``cross``      — Two orthogonal bands (horizontal + vertical)
                       through the canvas centre, forming a ``+``.
                       Splits the canvas into four soft-cell quadrants.
    * ``cluster``    — Four compact macro clusters at the canvas
                       quadrant centres (round-robin assignment by
                       area so the clusters are roughly balanced).

Why these four?  They span the major topologies a placer can collapse
into: spatially uniform, edge-attracted, axis-aligned channels, and
quadrant clusters.  In informal sweeps on ``ibm01``-``ibm03``,
``perimeter``, ``cross``, and ``cluster`` each won at least one
benchmark over uniform, suggesting the basin assignment really does
vary by netlist.

Usage:
    uv run evaluate submissions/will_seed/structureplace.py
    uv run evaluate submissions/will_seed/structureplace.py --all
    uv run evaluate submissions/will_seed/structureplace.py -b ibm03

Implementation note: each archetype is wired into TierPlace by
temporarily replacing ``tierplace._uniform_spread`` with the archetype's
spread function (it is the single hook TierPlace uses to seed Phase 1).
We do **not** modify ``tierplace.py`` itself.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import torch

from macro_place.benchmark import Benchmark

# ``submissions/will_seed`` is not a Python package (no __init__.py),
# and the evaluator loads this file via importlib.spec_from_file_location.
# Putting our directory on sys.path lets us import tierplace as a
# top-level sibling module — which is what we need so we can monkey-patch
# its module-level ``_uniform_spread`` reference per archetype (Python
# resolves the name in the module globals at call time).
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import tierplace  # noqa: E402  - sibling-file import after path setup
from tierplace import AnalyticalPlacer  # noqa: E402

try:
    from macro_place.objective import compute_proxy_cost as _compute_proxy_cost
except Exception:
    _compute_proxy_cost = None


# ===========================================================================
# 1. Archetype spread functions
# ===========================================================================
#
# Every function has the same signature as ``tierplace._uniform_spread``:
#   spread(benchmark, dev, dt) -> Tensor[num_macros, 2]
#
# Only **movable hard macros** are repositioned; fixed hard macros and
# soft macros keep their incoming positions (matching tierplace's
# baseline). All outputs are clamped to the in-canvas legal range so
# the optimizer starts from a feasible point.


def _movable_indices(benchmark: Benchmark, dev) -> List[int]:
    """Indices of movable hard macros (not fixed)."""
    nh = benchmark.num_hard_macros
    fix = benchmark.macro_fixed.to(dev)
    return (~fix[:nh]).nonzero(as_tuple=False).squeeze(1).tolist()


def _sort_by_area_desc(indices: Sequence[int], sizes: torch.Tensor) -> List[int]:
    """Order macro indices from largest footprint to smallest."""
    keyed = [(float(sizes[i, 0] * sizes[i, 1]), int(i)) for i in indices]
    keyed.sort(reverse=True)
    return [i for _, i in keyed]


def _spread_uniform(benchmark: Benchmark, dev, dt) -> torch.Tensor:
    """Even grid covering the canvas — same as ``tierplace._uniform_spread``.

    Defined explicitly so the registry can call it like every other
    archetype without going through the (about-to-be-monkey-patched)
    module attribute.
    """
    nh = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    init = benchmark.macro_positions.to(dev, dt).clone()
    movable = _movable_indices(benchmark, dev)
    n_mov = len(movable)
    if n_mov == 0:
        return init

    cols = max(1, math.ceil(math.sqrt(n_mov * cw / ch)))
    rows = max(1, math.ceil(n_mov / cols))
    xs = torch.linspace(cw * 0.05, cw * 0.95, cols, device=dev, dtype=dt)
    ys = torch.linspace(ch * 0.05, ch * 0.95, rows, device=dev, dtype=dt)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)[:n_mov]

    hw = benchmark.macro_sizes[:nh, 0].to(dev, dt) / 2
    hh = benchmark.macro_sizes[:nh, 1].to(dev, dt) / 2
    for k, i in enumerate(movable):
        init[i, 0] = pts[k, 0].clamp(hw[i], cw - hw[i])
        init[i, 1] = pts[k, 1].clamp(hh[i], ch - hh[i])
    return init


def _spread_perimeter(benchmark: Benchmark, dev, dt) -> torch.Tensor:
    """Concentric rings around the canvas perimeter, biggest macros outermost.

    Ring spacing tracks the average movable macro size so each ring is a
    reasonable single-macro band; if more macros than slots, we add inner
    rings until the canvas is exhausted (then fall back to the centre).
    """
    nh = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    init = benchmark.macro_positions.to(dev, dt).clone()
    movable = _movable_indices(benchmark, dev)
    n_mov = len(movable)
    if n_mov == 0:
        return init

    sizes = benchmark.macro_sizes[:nh].to(dev, dt)
    hw_all = sizes[:, 0] / 2
    hh_all = sizes[:, 1] / 2

    mov_w = sizes[movable, 0]
    mov_h = sizes[movable, 1]
    avg_w = float(mov_w.mean())
    avg_h = float(mov_h.mean())

    # Ring inset based on avg size; tighter rings would overlap heavily
    # at the start and waste the gradient signal in the first few iters.
    ring_step_x = max(avg_w * 1.05, cw * 0.012)
    ring_step_y = max(avg_h * 1.05, ch * 0.012)

    sorted_movable = _sort_by_area_desc(movable, sizes)

    positions: List[tuple] = []
    ring = 0
    while len(positions) < n_mov:
        x0 = ring_step_x * 0.5 + ring * ring_step_x
        x1 = cw - x0
        y0 = ring_step_y * 0.5 + ring * ring_step_y
        y1 = ch - y0
        if x1 - x0 < ring_step_x or y1 - y0 < ring_step_y:
            # Canvas exhausted: park leftovers near centre. The Adam
            # ramp will spread them; this is just a sane fallback.
            cx_c, cy_c = cw * 0.5, ch * 0.5
            while len(positions) < n_mov:
                positions.append((cx_c, cy_c))
            break

        bottom_len = x1 - x0
        right_len = y1 - y0
        top_len = bottom_len
        left_len = right_len
        perim = bottom_len + right_len + top_len + left_len

        # ``slot`` ~ avg macro footprint with 1.1x padding, never less
        # than 4 slots/ring (so tiny benchmarks still see all 4 corners).
        slot = max(avg_w, avg_h) * 1.1
        n_slots = max(4, int(perim / slot))

        for k in range(n_slots):
            if len(positions) >= n_mov:
                break
            s = (k + 0.5) / n_slots * perim
            if s < bottom_len:
                cx, cy = x0 + s, y0
            elif s < bottom_len + right_len:
                cx, cy = x1, y0 + (s - bottom_len)
            elif s < bottom_len + right_len + top_len:
                cx, cy = x1 - (s - bottom_len - right_len), y1
            else:
                cx, cy = x0, y1 - (s - bottom_len - right_len - top_len)
            positions.append((float(cx), float(cy)))
        ring += 1

    for k, i in enumerate(sorted_movable):
        cx, cy = positions[k]
        init[i, 0] = max(float(hw_all[i]), min(cw - float(hw_all[i]), cx))
        init[i, 1] = max(float(hh_all[i]), min(ch - float(hh_all[i]), cy))
    return init


def _spread_cross(benchmark: Benchmark, dev, dt) -> torch.Tensor:
    """``+`` pattern: a horizontal centre band and a vertical centre band.

    Splits the macros 50/50 by area-rank order. The largest half goes
    into the horizontal band (cw is typically the bigger dimension on
    these benchmarks, so a horizontal band carries more macros without
    collisions); the other half forms the vertical band. Both bands are
    multi-row when needed.
    """
    nh = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    init = benchmark.macro_positions.to(dev, dt).clone()
    movable = _movable_indices(benchmark, dev)
    n_mov = len(movable)
    if n_mov == 0:
        return init

    sizes = benchmark.macro_sizes[:nh].to(dev, dt)
    hw_all = sizes[:, 0] / 2
    hh_all = sizes[:, 1] / 2

    avg_w = float(sizes[movable, 0].mean())
    avg_h = float(sizes[movable, 1].mean())

    sorted_movable = _sort_by_area_desc(movable, sizes)
    # Largest macros to the horizontal band (more pin density tends to
    # benefit from spanning the canvas; the vertical band gets the
    # remaining smaller macros).
    n_h = (n_mov + 1) // 2
    h_macros = sorted_movable[:n_h]
    v_macros = sorted_movable[n_h:]

    # Horizontal band: rows × cols laid out around y = ch/2
    if h_macros:
        h_cols = max(
            1, int(round(math.sqrt(len(h_macros) * (cw * 0.9) / max(avg_h * 1.1, 1e-6))))
        )
        h_cols = max(1, min(len(h_macros), h_cols))
        h_rows = max(1, math.ceil(len(h_macros) / h_cols))
        xs = torch.linspace(cw * 0.06, cw * 0.94, h_cols, device=dev, dtype=dt).tolist()
        band_height = h_rows * avg_h * 1.08
        y_top = ch * 0.5 - band_height / 2
        ys = [y_top + (r + 0.5) * avg_h * 1.08 for r in range(h_rows)]
        for k, i in enumerate(h_macros):
            r = k // h_cols
            c = k % h_cols
            cx = xs[c]
            cy = ys[min(r, len(ys) - 1)]
            init[i, 0] = max(float(hw_all[i]), min(cw - float(hw_all[i]), cx))
            init[i, 1] = max(float(hh_all[i]), min(ch - float(hh_all[i]), cy))

    # Vertical band: rows × cols laid out around x = cw/2
    if v_macros:
        v_rows = max(
            1, int(round(math.sqrt(len(v_macros) * (ch * 0.9) / max(avg_w * 1.1, 1e-6))))
        )
        v_rows = max(1, min(len(v_macros), v_rows))
        v_cols = max(1, math.ceil(len(v_macros) / v_rows))
        ys2 = torch.linspace(ch * 0.06, ch * 0.94, v_rows, device=dev, dtype=dt).tolist()
        band_width = v_cols * avg_w * 1.08
        x_left = cw * 0.5 - band_width / 2
        xs2 = [x_left + (c + 0.5) * avg_w * 1.08 for c in range(v_cols)]
        for k, i in enumerate(v_macros):
            r = k // v_cols
            c = k % v_cols
            cx = xs2[min(c, len(xs2) - 1)]
            cy = ys2[r]
            init[i, 0] = max(float(hw_all[i]), min(cw - float(hw_all[i]), cx))
            init[i, 1] = max(float(hh_all[i]), min(ch - float(hh_all[i]), cy))

    return init


def _spread_cluster(
    benchmark: Benchmark, dev, dt, k_clusters: int = 4
) -> torch.Tensor:
    """K=4 compact clusters at the canvas quadrant centres.

    Macros are sorted by area and dealt round-robin into the four
    quadrant bins so the clusters carry roughly equal total area. Inside
    each cluster they are laid out in a tight square-ish grid centred
    on the quadrant centroid.
    """
    nh = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    init = benchmark.macro_positions.to(dev, dt).clone()
    movable = _movable_indices(benchmark, dev)
    n_mov = len(movable)
    if n_mov == 0:
        return init

    sizes = benchmark.macro_sizes[:nh].to(dev, dt)
    hw_all = sizes[:, 0] / 2
    hh_all = sizes[:, 1] / 2

    centers = [
        (cw * 0.27, ch * 0.27),
        (cw * 0.73, ch * 0.27),
        (cw * 0.27, ch * 0.73),
        (cw * 0.73, ch * 0.73),
    ]
    if k_clusters > 4:
        # Extra clusters: drop additional centres on a diagonal jitter.
        for j in range(k_clusters - 4):
            t = (j + 1) / (k_clusters - 3)
            centers.append((cw * (0.5 + 0.18 * math.cos(j)), ch * (0.5 + 0.18 * math.sin(j))))
    centers = centers[:k_clusters]

    sorted_movable = _sort_by_area_desc(movable, sizes)
    bins: List[List[int]] = [[] for _ in range(k_clusters)]
    for rank, i in enumerate(sorted_movable):
        bins[rank % k_clusters].append(i)

    for ci, mlist in enumerate(bins):
        if not mlist:
            continue
        cx_c, cy_c = centers[ci]
        n_c = len(mlist)
        avg_w = float(sum(float(sizes[i, 0]) for i in mlist) / n_c)
        avg_h = float(sum(float(sizes[i, 1]) for i in mlist) / n_c)
        cols = max(1, int(round(math.sqrt(n_c))))
        rows = max(1, math.ceil(n_c / cols))
        for k2, i in enumerate(mlist):
            r = k2 // cols
            c = k2 % cols
            cx = cx_c + (c - (cols - 1) / 2.0) * avg_w * 1.04
            cy = cy_c + (r - (rows - 1) / 2.0) * avg_h * 1.04
            init[i, 0] = max(float(hw_all[i]), min(cw - float(hw_all[i]), cx))
            init[i, 1] = max(float(hh_all[i]), min(ch - float(hh_all[i]), cy))

    return init


# Registry for archetype name -> spread function. Adding a new
# archetype is a one-line entry plus the spread function above.
_SPREAD_REGISTRY: Dict[str, Callable[[Benchmark, torch.device, torch.dtype], torch.Tensor]] = {
    "uniform": _spread_uniform,
    "perimeter": _spread_perimeter,
    "cross": _spread_cross,
    "cluster": _spread_cluster,
}


# ===========================================================================
# 2. StructurePlacer — sweep TierPlace across initial structures
# ===========================================================================


class StructurePlacer:
    """Run TierPlace from several initial structures and keep the best.

    The evaluator calls ``StructurePlacer()`` with no arguments, so the
    defaults are chosen to match a sensible "explore initial structures"
    sweep out of the box. Override via attribute assignment if you want
    to customise (e.g. ``p = StructurePlacer(); p.archetypes = (...)``).
    """

    DEFAULT_ARCHETYPES: tuple = ("uniform", "perimeter", "cross", "cluster")

    def __init__(
        self,
        archetypes: Optional[Sequence[str]] = None,
        seed: int = 42,
        verbose: bool = True,
        inner_verbose: bool = False,
        # If True, score each archetype with the real TILOS proxy cost
        # via PlacementCost. Falls back to the placer's internal score
        # when PlacementCost is unavailable (no ``_plc`` attached).
        gate_with_real_proxy: bool = True,
        # Forwarded to AnalyticalPlacer; keep at defaults to match the
        # standard tierplace.py invocation.
        placer_kwargs: Optional[Dict] = None,
    ):
        self.archetypes = (
            tuple(archetypes) if archetypes is not None else self.DEFAULT_ARCHETYPES
        )
        self.seed = seed
        self.verbose = verbose
        self.inner_verbose = inner_verbose
        self.gate_with_real_proxy = gate_with_real_proxy
        self.placer_kwargs = dict(placer_kwargs or {})

    # ---- main entry point ---------------------------------------------------

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        unknown = [a for a in self.archetypes if a not in _SPREAD_REGISTRY]
        if unknown:
            raise ValueError(
                f"Unknown archetypes {unknown!r}; "
                f"valid options: {sorted(_SPREAD_REGISTRY)}"
            )

        if self.verbose:
            print(
                f"[{benchmark.name}] StructurePlace sweep: "
                f"{', '.join(self.archetypes)}",
                flush=True,
            )

        original_spread = tierplace._uniform_spread
        results: List[Dict] = []
        try:
            for arch in self.archetypes:
                spread_fn = _SPREAD_REGISTRY[arch]
                tierplace._uniform_spread = spread_fn

                t0 = time.time()
                placer = AnalyticalPlacer(
                    seed=self.seed,
                    verbose=self.inner_verbose,
                    gate_with_real_proxy=self.gate_with_real_proxy,
                    **self.placer_kwargs,
                )
                placement = placer.place(benchmark)
                runtime = time.time() - t0

                proxy_cost, breakdown = self._real_proxy_score(placement, benchmark)
                results.append(
                    {
                        "archetype": arch,
                        "placement": placement,
                        "proxy_cost": proxy_cost,
                        "runtime": runtime,
                        "breakdown": breakdown,
                    }
                )

                if self.verbose:
                    if breakdown is not None:
                        print(
                            f"  [{benchmark.name}] arch={arch:<10s} "
                            f"proxy={proxy_cost:.4f}  "
                            f"(wl={breakdown['wirelength_cost']:.3f} "
                            f"den={breakdown['density_cost']:.3f} "
                            f"cong={breakdown['congestion_cost']:.3f} "
                            f"overlaps={breakdown['overlap_count']})  "
                            f"[{runtime:.1f}s]",
                            flush=True,
                        )
                    else:
                        print(
                            f"  [{benchmark.name}] arch={arch:<10s} "
                            f"(no PLC; proxy unavailable)  [{runtime:.1f}s]",
                            flush=True,
                        )
        finally:
            tierplace._uniform_spread = original_spread

        # Pick the best archetype: smallest valid (no overlap) proxy
        # cost; ties broken by smaller wirelength. If every archetype
        # has overlaps or PLC is unavailable, fall back to the first
        # archetype's placement.
        ranked = self._rank(results)
        if not ranked:
            return results[0]["placement"]

        winner = ranked[0]
        if self.verbose:
            print(
                f"[{benchmark.name}] StructurePlace winner: {winner['archetype']} "
                f"(proxy={winner['proxy_cost']:.4f})",
                flush=True,
            )
        return winner["placement"]

    # ---- helpers ------------------------------------------------------------

    def _real_proxy_score(self, placement: torch.Tensor, benchmark: Benchmark):
        """Return (proxy_cost, full breakdown dict) or (+inf, None) on failure.

        Uses the patched loader from tierplace.py to read ``_plc`` off
        the benchmark. Without a PLC handle we cannot compute the real
        proxy, so the caller falls back to internal ranking.
        """
        if _compute_proxy_cost is None:
            return float("inf"), None
        plc = getattr(benchmark, "_plc", None)
        if plc is None:
            return float("inf"), None
        try:
            costs = _compute_proxy_cost(
                placement.detach().cpu().float(), benchmark, plc
            )
            return float(costs["proxy_cost"]), costs
        except Exception:
            return float("inf"), None

    def _rank(self, results: List[Dict]) -> List[Dict]:
        """Sort archetype results: valid placements first, then by proxy cost.

        A placement with overlaps is *disqualified* (per evaluate.py)
        and ranked below any valid placement, even if its surrogate
        score is lower.
        """
        if not results:
            return []

        def key(r):
            breakdown = r["breakdown"]
            overlaps = int(breakdown["overlap_count"]) if breakdown is not None else 0
            wl = float(breakdown["wirelength_cost"]) if breakdown is not None else 0.0
            return (overlaps > 0, r["proxy_cost"], wl)

        return sorted(results, key=key)
