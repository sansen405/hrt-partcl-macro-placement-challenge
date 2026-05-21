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
    * ``cross``      — Four dense quadrant blocks centred at
                       ``(W/4, H/4)`` etc., leaving a clean ``+`` of
                       empty space through the canvas centre and a
                       margin around the canvas edges.
    * ``cluster``    — Four dense quadrant blocks anchored at the
                       canvas corners, growing inward.  Empty space
                       only appears as a ``+`` through the middle —
                       the blocks themselves reach the outer edges.

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
    """``+`` channels: four dense quadrant blocks centred in each quadrant.

    Macros are dealt round-robin (area-sorted) into four quadrant bins so
    the bins carry roughly equal total area. Inside each quadrant, the
    bin is laid out as a tight ``rows × cols`` grid centred on the
    quadrant centroid ``(W/4, H/4)`` etc.  Because each block is centred
    *inside* its quadrant, there is a margin on every side — including
    against the canvas edges and against the canvas centreline — which
    leaves a clean ``+`` of empty space through the middle for soft
    cells to flow through.
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
        (cw * 0.25, ch * 0.25),
        (cw * 0.75, ch * 0.25),
        (cw * 0.25, ch * 0.75),
        (cw * 0.75, ch * 0.75),
    ]

    sorted_movable = _sort_by_area_desc(movable, sizes)
    bins: List[List[int]] = [[] for _ in range(4)]
    for rank, i in enumerate(sorted_movable):
        bins[rank % 4].append(i)

    for ci, mlist in enumerate(bins):
        if not mlist:
            continue
        cx_c, cy_c = centers[ci]
        n_c = len(mlist)
        avg_w = float(sum(float(sizes[i, 0]) for i in mlist) / n_c)
        avg_h = float(sum(float(sizes[i, 1]) for i in mlist) / n_c)
        cols = max(1, int(round(math.sqrt(n_c))))
        rows = max(1, math.ceil(n_c / cols))
        # Tight pitch so the block stays compact and the ``+`` channel
        # between blocks is wide and clearly visible.
        pitch_w = avg_w * 1.04
        pitch_h = avg_h * 1.04
        for k2, i in enumerate(mlist):
            r = k2 // cols
            c = k2 % cols
            cx = cx_c + (c - (cols - 1) / 2.0) * pitch_w
            cy = cy_c + (r - (rows - 1) / 2.0) * pitch_h
            init[i, 0] = max(float(hw_all[i]), min(cw - float(hw_all[i]), cx))
            init[i, 1] = max(float(hh_all[i]), min(ch - float(hh_all[i]), cy))

    return init


def _spread_cluster(
    benchmark: Benchmark, dev, dt, k_clusters: int = 4
) -> torch.Tensor:
    """Four corner-anchored quadrant clusters that grow inward.

    Each of the four quadrants gets a tight grid of macros that is
    flush with the canvas corner and expands toward the canvas centre.
    Compared to ``cross`` (whose blocks are centred in their quadrants
    and therefore have a margin on *all* sides), ``cluster`` only has a
    margin on the inner sides of each block.  The empty space therefore
    forms a single ``+`` through the middle of the canvas while the
    blocks themselves reach the outer edges.

    Macros are sorted by area and dealt round-robin into the four
    quadrant bins so each cluster carries roughly equal total area.
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

    sorted_movable = _sort_by_area_desc(movable, sizes)
    k_eff = max(1, min(k_clusters, 4))
    bins: List[List[int]] = [[] for _ in range(k_eff)]
    for rank, i in enumerate(sorted_movable):
        bins[rank % k_eff].append(i)

    # Each quadrant is identified by a corner anchor and an inward
    # direction.  Anchor (ax, ay) is the quadrant's outer canvas
    # corner; direction (dx, dy) points toward the canvas centre.
    quadrants = [
        ((0.0, 0.0), (+1, +1)),         # top-left  (anchor at canvas top-left)
        ((cw, 0.0), (-1, +1)),          # top-right
        ((0.0, ch), (+1, -1)),          # bottom-left
        ((cw, ch), (-1, -1)),           # bottom-right
    ][:k_eff]

    # Inner margin (toward the canvas centre) — this is what creates
    # the visible ``+`` channel between the four blocks.
    inner_margin_x = cw * 0.04
    inner_margin_y = ch * 0.04

    for ci, mlist in enumerate(bins):
        if not mlist:
            continue
        (ax, ay), (dx, dy) = quadrants[ci]
        n_c = len(mlist)
        avg_w = float(sum(float(sizes[i, 0]) for i in mlist) / n_c)
        avg_h = float(sum(float(sizes[i, 1]) for i in mlist) / n_c)
        cols = max(1, int(round(math.sqrt(n_c))))
        rows = max(1, math.ceil(n_c / cols))
        pitch_w = avg_w * 1.04
        pitch_h = avg_h * 1.04
        # Place macro k2 at column c, row r counted *outward* from the
        # corner anchor. The first macro centre sits half a tile away
        # from the canvas edge so the block lies fully on-canvas.
        for k2, i in enumerate(mlist):
            r = k2 // cols
            c = k2 % cols
            cx = ax + dx * (avg_w * 0.5 + c * pitch_w)
            cy = ay + dy * (avg_h * 0.5 + r * pitch_h)
            # Keep the block on its own side of the centreline so the
            # ``+`` channel stays clean even if rows/cols overflow.
            if dx > 0:
                cx = min(cx, cw * 0.5 - inner_margin_x - float(hw_all[i]))
            else:
                cx = max(cx, cw * 0.5 + inner_margin_x + float(hw_all[i]))
            if dy > 0:
                cy = min(cy, ch * 0.5 - inner_margin_y - float(hh_all[i]))
            else:
                cy = max(cy, ch * 0.5 + inner_margin_y + float(hh_all[i]))
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
    # Halo sweep for each archetype. Different halo_frac values give
    # Phase 1 different effective macro footprints, which lands the
    # global Adam in different basins → diversification along an axis
    # orthogonal to the structural archetype itself.
    DEFAULT_HALOS: tuple = (0.04, 0.06, 0.08, 0.10)

    def __init__(
        self,
        archetypes: Optional[Sequence[str]] = None,
        halos: Optional[Sequence[float]] = None,
        seed: int = 42,
        verbose: bool = True,
        inner_verbose: bool = False,
        # If True, score each archetype with the real TILOS proxy cost
        # via PlacementCost. Falls back to the placer's internal score
        # when PlacementCost is unavailable (no ``_plc`` attached).
        gate_with_real_proxy: bool = True,
        # Forwarded to AnalyticalPlacer; keep at defaults to match the
        # standard tierplace.py invocation. ``halo_frac`` here is
        # overridden by the per-run halo sweep value.
        placer_kwargs: Optional[Dict] = None,
    ):
        self.archetypes = (
            tuple(archetypes) if archetypes is not None else self.DEFAULT_ARCHETYPES
        )
        self.halos = (
            tuple(halos) if halos is not None else self.DEFAULT_HALOS
        )
        self.seed = seed
        self.verbose = verbose
        self.inner_verbose = inner_verbose
        self.gate_with_real_proxy = gate_with_real_proxy
        # Strip halo_frac from placer_kwargs — the sweep controls it.
        self.placer_kwargs = {
            k: v for k, v in dict(placer_kwargs or {}).items() if k != "halo_frac"
        }

    # ---- main entry point ---------------------------------------------------

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        unknown = [a for a in self.archetypes if a not in _SPREAD_REGISTRY]
        if unknown:
            raise ValueError(
                f"Unknown archetypes {unknown!r}; "
                f"valid options: {sorted(_SPREAD_REGISTRY)}"
            )

        n_runs = len(self.archetypes) * len(self.halos)
        if self.verbose:
            print(
                f"[{benchmark.name}] StructurePlace sweep: "
                f"{len(self.archetypes)} archetypes × {len(self.halos)} halos "
                f"= {n_runs} runs "
                f"(archetypes={', '.join(self.archetypes)}; "
                f"halos={', '.join(f'{h:.2f}' for h in self.halos)})",
                flush=True,
            )

        original_spread = tierplace._uniform_spread
        results: List[Dict] = []
        try:
            for arch in self.archetypes:
                spread_fn = _SPREAD_REGISTRY[arch]
                tierplace._uniform_spread = spread_fn

                for halo in self.halos:
                    t0 = time.time()
                    placer = AnalyticalPlacer(
                        seed=self.seed,
                        verbose=self.inner_verbose,
                        gate_with_real_proxy=self.gate_with_real_proxy,
                        halo_frac=halo,
                        **self.placer_kwargs,
                    )
                    placement = placer.place(benchmark)
                    runtime = time.time() - t0

                    proxy_cost, breakdown = self._real_proxy_score(
                        placement, benchmark
                    )
                    results.append(
                        {
                            "archetype": arch,
                            "halo": halo,
                            "placement": placement,
                            "proxy_cost": proxy_cost,
                            "runtime": runtime,
                            "breakdown": breakdown,
                        }
                    )

                    if self.verbose:
                        tag = f"{arch}@h={halo:.2f}"
                        if breakdown is not None:
                            print(
                                f"  [{benchmark.name}] {tag:<22s} "
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
                                f"  [{benchmark.name}] {tag:<22s} "
                                f"(no PLC; proxy unavailable)  [{runtime:.1f}s]",
                                flush=True,
                            )
        finally:
            tierplace._uniform_spread = original_spread

        # Pick the best (archetype, halo) pair: smallest valid (no
        # overlap) proxy cost; ties broken by smaller wirelength. If
        # every run has overlaps or PLC is unavailable, fall back to
        # the first run's placement.
        ranked = self._rank(results)
        if not ranked:
            return results[0]["placement"]

        winner = ranked[0]
        if self.verbose:
            print(
                f"[{benchmark.name}] StructurePlace winner: "
                f"{winner['archetype']}@h={winner['halo']:.2f} "
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
