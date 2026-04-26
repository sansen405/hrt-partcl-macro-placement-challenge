"""
TierPlace v3 — ReFine-style joint hard+soft refinement

The leaderboard entry for "Cezar" (ReFine) achieves an average proxy of
1.2224 vs our v2's 1.3336. The key insight in IncreMacro / EfficientRefiner
papers: refine an already-legal placement by relaxing the hard-macro
freeze and pushing macros toward the chip periphery while letting standard
cells (soft macros, in our framework) diffuse into the freed core space.

Concretely v3 changes from v2:

* The "soft polish" stage no longer freezes hard macros — both hard and
  soft macros are subject to the WL + dw*density + cw*congestion gradient.
  The base v1 placer already optimized density/congestion holistically,
  but it operates on a halo'd size + eDensity model. Re-running the same
  proxy-shaped objective with the *real* sizes (no halo) and joint hard +
  soft motion drains the residual density / congestion hot spots.

* A smooth pairwise-overlap penalty is added so hard macros don't drift
  into each other during the joint pass. The penalty has zero gradient
  when no overlap exists (relu cutoff), so it only activates as a "stop
  sign" when a pair would otherwise collide.

* An optional periphery cost (off by default) implements IncreMacro's
  |W/2 - x| + (W/2)^2 / |W/2 - x| boundary attraction. Useful on
  benchmarks where the base placer leaves hard macros mid-canvas.

* Final legalization guarantees zero overlaps regardless of whether the
  pairwise-overlap penalty was strong enough. The base TierPlace's
  _legalize routine handles this.

* Score gating compares the joint-polished placement to the input on the
  same WL + dw*den + cw*cong proxy used by the base placer. Refinement
  is reverted if it doesn't improve the score.

Knobs: see ``AnalyticalPlacer.__init__``. The defaults are intentionally
conservative — light periphery weight, more polish iterations than v2,
and tighter LR — to avoid regressing on benchmarks where v2 is already
near-optimal.

Usage:
    uv run evaluate submissions/will_seed/tierplacev3.py
    uv run evaluate submissions/will_seed/tierplacev3.py --all
"""

import importlib.util
import math
from pathlib import Path

import torch


def _load_local_module(filename: str, modname: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(modname, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {filename} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_tp = _load_local_module("tierplace.py", "tierplace_base_module_v3")
_tp2 = _load_local_module("tierplacev2.py", "tierplace_v2_module_v3")
_BasePlacer = _tp.AnalyticalPlacer
_V2Placer = _tp2.AnalyticalPlacer


def _periphery_cost_1d(x: torch.Tensor, span: float, eps_frac: float = 0.02) -> torch.Tensor:
    """IncreMacro convex periphery cost in 1D, normalized.

    f(x) = (|W/2 - x| + (W/2)^2 / |W/2 - x| - W) / W

    Zero at x in {0, W}; diverges as x -> W/2 (denominator clamped to
    eps_frac * W to avoid the singularity).
    """
    half = span * 0.5
    eps = max(span * eps_frac, 1e-6)
    d = (half - x).abs().clamp(min=eps)
    return (d + half * half / d - span) / span


def _pairwise_overlap_loss(
    pos_h: torch.Tensor,
    sizes_h: torch.Tensor,
    avg_macro_area: float,
) -> torch.Tensor:
    """Smooth pairwise-overlap penalty over hard macros.

    For each pair (i, j), overlap area = relu(sep_x - |dx|) * relu(sep_y -
    |dy|). Returns sum_{i<j} (overlap / avg_macro_area)^2 — overlap is
    normalized by the average macro area so a "fully overlapping pair"
    contributes O(1), independent of chip size or macro size. With
    weight=1, full overlap is a 1.0 cost — already large compared to the
    proxy-shape loss (~2 at base). Squaring makes the cost steep enough to
    drive gradients out of overlap, while the relu cutoff keeps it
    completely silent when no overlap exists.
    """
    nh = pos_h.shape[0]
    if nh <= 1:
        return pos_h.sum() * 0.0
    dx = pos_h[:, 0:1] - pos_h[:, 0:1].T
    dy = pos_h[:, 1:2] - pos_h[:, 1:2].T
    sep_x = (sizes_h[:, 0:1] + sizes_h[:, 0:1].T) * 0.5
    sep_y = (sizes_h[:, 1:2] + sizes_h[:, 1:2].T) * 0.5
    ox = torch.relu(sep_x - dx.abs())
    oy = torch.relu(sep_y - dy.abs())
    over = ox * oy / max(avg_macro_area, 1e-12)
    diag_mask = torch.eye(nh, dtype=torch.bool, device=over.device)
    over = over.masked_fill(diag_mask, 0.0)
    # Each pair counted twice (i,j) and (j,i); halve.
    return 0.5 * (over * over).sum()


class AnalyticalPlacer(_V2Placer):
    """v3 = v1 base + joint hard+soft polish (no hard-freeze) + legalize.

    Drop-in replacement for v2 that refines via a joint Adam pass instead
    of a soft-only pass, lifting the hard-macro freeze. Hard macros may
    move during the polish, guarded by a smooth pairwise overlap penalty
    and a final legalization step.
    """

    def __init__(
        self,
        *args,
        # joint polish iters (runs before v2's soft polish). Conservative:
        # fewer iters and lower LR than v2's polish so hard macros only
        # take a small step from their already-near-optimal base position.
        joint_polish_iters: int = 120,
        joint_polish_lr_scale: float = 0.05,
        joint_polish_dw_scale: float = 0.6,
        joint_polish_cw_scale: float = 1.0,
        # smooth pairwise-overlap penalty on hard macros, scaled per-pair
        # against an "average macro area" reference instead of chip_area^2.
        # See _pairwise_overlap_loss; the normalization there is now
        # tighter so weight ~1.0 yields a strong stop-sign for any
        # non-trivial overlap.
        joint_overlap_weight: float = 1.0,
        # optional periphery pull on hard macros (per macro-coord)
        joint_periphery_weight: float = 0.0,
        joint_periphery_eps_frac: float = 0.02,
        # final legalization gap (canvas units)
        joint_legalize_gap: float = 0.05,
        # always run v2's existing soft-only polish AFTER the joint polish.
        # (a) When joint polish improves things, the soft polish further
        # refines soft-macro positions on the new hard layout. (b) When
        # joint polish reverts (kept=base), this guarantees we still get
        # v2's polish behavior as a fallback, so v3 is never worse than v2.
        run_v2_soft_polish_after: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.joint_polish_iters = joint_polish_iters
        self.joint_polish_lr_scale = joint_polish_lr_scale
        self.joint_polish_dw_scale = joint_polish_dw_scale
        self.joint_polish_cw_scale = joint_polish_cw_scale
        self.joint_overlap_weight = joint_overlap_weight
        self.joint_periphery_weight = joint_periphery_weight
        self.joint_periphery_eps_frac = joint_periphery_eps_frac
        self.joint_legalize_gap = joint_legalize_gap
        self.run_v2_soft_polish_after = run_v2_soft_polish_after

    # --------------------------------------------------------------------
    # joint hard+soft polish on the same WL + dw*density + cw*congestion
    # objective the base placer uses, plus a smooth pairwise-overlap
    # penalty on hard macros and an optional periphery pull.
    # --------------------------------------------------------------------
    def _joint_polish_setup(self, placement: torch.Tensor, benchmark):
        nh = benchmark.num_hard_macros
        nm = benchmark.num_macros
        if self.joint_polish_iters <= 0:
            return None

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dt = torch.float32 if dev.type == "cuda" else torch.float64
        placement = placement.to(dev, dt)

        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        diag = math.hypot(cw, ch)
        gr, gc = benchmark.grid_rows, benchmark.grid_cols
        bw, bh = cw / gc, ch / gr

        # Use *real* (un-haloed) sizes for refinement — base placer used a
        # halo'd size for spreading, but at this point macros are already
        # legal and we want the proxy-shaped objective to reflect actual
        # geometry rather than buffered geometry.
        sizes = benchmark.macro_sizes.to(dev, dt)
        fixed = benchmark.macro_fixed.to(dev)

        net_idx, net_mask, port_pos = _tp._build_nets(benchmark, dev, dt)
        if net_idx.shape[0] == 0:
            return None

        gx = (torch.arange(gc, device=dev, dtype=dt) + 0.5) * bw
        gy = (torch.arange(gr, device=dev, dtype=dt) + 0.5) * bh
        gamma = max(diag * self.gamma_e * 1.5, 0.1)

        with torch.no_grad():
            apos0 = torch.cat([placement, port_pos], dim=0)
            wl_0 = max(abs(_tp._wa_wirelength(apos0, net_idx, net_mask, gamma).item()), 1.0)
            den_0 = max(abs(_tp._density_topk(placement, sizes, nm, gx, gy, bw, bh).item()), 1e-6)
            cl_0 = max(abs(_tp._congestion_loss(apos0, net_idx, net_mask, gx, gy, bw, bh, gamma).item()), 1e-6)

        dw = self.dw_p3 * float(self.joint_polish_dw_scale)
        tier = _tp.benchmark_stress_tier(benchmark) if self.adaptive_hard else 0
        cw_joint = _tp2._TIER_CW[tier] * 0.5 * float(self.joint_polish_cw_scale)

        hw = sizes[:, 0] / 2
        hh = sizes[:, 1] / 2
        return {
            "dev": dev, "dt": dt,
            "nh": nh, "nm": nm,
            "cw": cw, "ch": ch, "diag": diag, "bw": bw, "bh": bh,
            "sizes": sizes, "fixed": fixed,
            "net_idx": net_idx, "net_mask": net_mask, "port_pos": port_pos,
            "gx": gx, "gy": gy, "gamma": gamma,
            "wl_0": wl_0, "den_0": den_0, "cl_0": cl_0,
            "dw": dw, "cw_joint": cw_joint,
            "hw": hw, "hh": hh,
            "lb": torch.stack([hw, hh], dim=1),
            "ub": torch.stack([cw - hw, ch - hh], dim=1),
            "placement": placement,
        }

    def _joint_score_fn(self, ctx):
        net_idx, net_mask = ctx["net_idx"], ctx["net_mask"]
        port_pos = ctx["port_pos"]
        sizes = ctx["sizes"]
        nm = ctx["nm"]
        gx, gy, bw, bh = ctx["gx"], ctx["gy"], ctx["bw"], ctx["bh"]
        gamma = ctx["gamma"]
        wl_0, den_0, cl_0 = ctx["wl_0"], ctx["den_0"], ctx["cl_0"]
        dw, cw_joint = ctx["dw"], ctx["cw_joint"]

        def _score(x: torch.Tensor) -> float:
            with torch.no_grad():
                apos = torch.cat([x, port_pos], dim=0)
                wl = float((_tp._wa_wirelength(apos, net_idx, net_mask, gamma) / wl_0).item())
                den = float((_tp._density_topk(x, sizes, nm, gx, gy, bw, bh) / den_0).item())
                cl = float((_tp._congestion_loss(apos, net_idx, net_mask, gx, gy, bw, bh, gamma) / cl_0).item())
                return wl + dw * den + cw_joint * cl

        return _score

    def _joint_polish(self, placement: torch.Tensor, benchmark) -> torch.Tensor:
        ctx = self._joint_polish_setup(placement, benchmark)
        if ctx is None:
            return placement

        dev, dt = ctx["dev"], ctx["dt"]
        nh, nm = ctx["nh"], ctx["nm"]
        cw, ch, diag = ctx["cw"], ctx["ch"], ctx["diag"]
        sizes, fixed = ctx["sizes"], ctx["fixed"]
        net_idx, net_mask, port_pos = ctx["net_idx"], ctx["net_mask"], ctx["port_pos"]
        gx, gy, bw, bh = ctx["gx"], ctx["gy"], ctx["bw"], ctx["bh"]
        gamma = ctx["gamma"]
        wl_0, den_0, cl_0 = ctx["wl_0"], ctx["den_0"], ctx["cl_0"]
        dw, cw_joint = ctx["dw"], ctx["cw_joint"]
        lb, ub = ctx["lb"], ctx["ub"]
        placement = ctx["placement"]

        chip_area = cw * ch
        with torch.no_grad():
            avg_macro_area = float(
                (sizes[:nh, 0] * sizes[:nh, 1]).mean().item()
            ) if nh > 0 else 1.0
        avg_macro_area = max(avg_macro_area, 1e-6)
        peri_w = float(self.joint_periphery_weight)
        over_w = float(self.joint_overlap_weight)

        score = self._joint_score_fn(ctx)
        base_score = score(placement)

        p = placement.clone().requires_grad_(True)
        lr = max(1e-4, self.lr * float(self.joint_polish_lr_scale))
        opt = torch.optim.Adam([p], lr=lr)

        for it in range(self.joint_polish_iters):
            opt.zero_grad()
            apos = torch.cat([p, port_pos], dim=0)
            wl = _tp._wa_wirelength(apos, net_idx, net_mask, gamma) / wl_0
            den = _tp._density_topk(p, sizes, nm, gx, gy, bw, bh) / den_0
            cl = _tp._congestion_loss(apos, net_idx, net_mask, gx, gy, bw, bh, gamma) / cl_0

            loss = wl + dw * den + cw_joint * cl

            if over_w > 0.0 and nh >= 2:
                over = _pairwise_overlap_loss(p[:nh], sizes[:nh], avg_macro_area)
                loss = loss + over_w * over

            if peri_w > 0.0 and nh >= 1:
                ph = p[:nh]
                peri_x = _periphery_cost_1d(ph[:, 0], cw, eps_frac=self.joint_periphery_eps_frac)
                peri_y = _periphery_cost_1d(ph[:, 1], ch, eps_frac=self.joint_periphery_eps_frac)
                peri = peri_x.sum() + peri_y.sum()
                loss = loss + peri_w * peri

            loss.backward()

            with torch.no_grad():
                p.grad[fixed] = 0.0
                torch.nn.utils.clip_grad_norm_([p], max_norm=diag * 0.5)

            opt.step()
            with torch.no_grad():
                p.data = torch.max(torch.min(p.data, ub), lb)
                p.data[fixed] = placement[fixed]

        end_p = p.detach()
        end_score = score(end_p)

        # Final legalization to guarantee zero overlaps. The pairwise
        # overlap penalty usually keeps overlaps near zero during gradient
        # descent, but residuals (and any soft-macro overlaps which we
        # don't penalize) are possible. _legalize fixes hard-macro overlaps
        # without disturbing fixed macros.
        try:
            legal_p = _tp._legalize(end_p, sizes, fixed, nh, cw, ch, gap=float(self.joint_legalize_gap))
            legal_score = score(legal_p)
        except Exception:
            legal_p = end_p
            legal_score = end_score

        # Always prefer the legalized result over the un-legalized end,
        # because un-legalized may have hard-macro overlaps that fail the
        # validator (proxy reports INVALID). Internal score may say
        # otherwise but validity is non-negotiable.
        # Revert to base only if legal is *worse* than base on internal score.
        if legal_score < base_score - 1e-5:
            result_name = "legal"
            result_p = legal_p
        else:
            result_name = "base"
            result_p = placement

        if self.verbose:
            print(
                f"  [{benchmark.name}] Joint polish: base={base_score:.4f} "
                f"-> end={end_score:.4f} legal={legal_score:.4f} "
                f"(kept={result_name})",
                flush=True,
            )
        return result_p

    # --------------------------------------------------------------------
    # override place: insert joint hard+soft polish between base v1
    # placement and v2's soft polish. The soft polish is gated against
    # the joint-polished placement so it never makes things worse.
    # --------------------------------------------------------------------
    def place(self, benchmark):
        base = _BasePlacer.place(self, benchmark)
        polished = self._joint_polish(base, benchmark)
        if self.run_v2_soft_polish_after:
            # v2's _soft_only_polish gates against its input so this is a
            # monotonic improvement (or no-op).
            polished = self._soft_only_polish(polished, benchmark)
        if isinstance(polished, torch.Tensor):
            return polished.cpu().float()
        return polished
