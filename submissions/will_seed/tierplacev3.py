"""TierPlace v3 — joint hard+soft refinement.

Lifts v2's hard-macro freeze: a WL + dw*density + cw*congestion gradient
pass moves both hard and soft macros, guarded by a smooth pairwise-overlap
penalty on hard macros and a final legalization step. Phase outputs are
gated on the actual TILOS proxy when available (loader attaches `_plc`),
falling back to the internal score otherwise.

Usage:
    uv run evaluate submissions/will_seed/tierplacev3.py
    uv run evaluate submissions/will_seed/tierplacev3.py --all
"""

import importlib.util
import math
from pathlib import Path

import torch

try:
    from macro_place.objective import compute_proxy_cost as _compute_proxy_cost
except Exception:
    _compute_proxy_cost = None


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


def _build_pin_world_index(macro_pin_offsets, num_hard, dev, dt):
    """Flatten per-macro pin offsets into (offsets[P,2], owner[P]) tensors."""
    if num_hard <= 0 or not macro_pin_offsets:
        return None, None
    flat_off, flat_owner = [], []
    for i in range(min(num_hard, len(macro_pin_offsets))):
        off_i = macro_pin_offsets[i]
        if off_i is None or off_i.numel() == 0:
            continue
        flat_off.append(off_i.to(device=dev, dtype=dt))
        flat_owner.append(torch.full((off_i.shape[0],), i, dtype=torch.long, device=dev))
    if not flat_off:
        return None, None
    return torch.cat(flat_off, dim=0), torch.cat(flat_owner, dim=0)


def _pin_density_topk(pos_h, pin_offsets, pin_owner, gx, gy, bw, bh, top_frac=0.05):
    """Top-K mean of soft hard-macro pin density on the routing grid."""
    if pin_offsets is None or pin_offsets.shape[0] == 0:
        return pos_h.sum() * 0.0
    px = pos_h[pin_owner, 0] + pin_offsets[:, 0]
    py = pos_h[pin_owner, 1] + pin_offsets[:, 1]
    tau = max(bw, bh) * 0.3
    sx = torch.sigmoid((px[:, None] - (gx - bw * 0.5)[None, :]) / tau) - torch.sigmoid(
        (px[:, None] - (gx + bw * 0.5)[None, :]) / tau
    )
    sy = torch.sigmoid((py[:, None] - (gy - bh * 0.5)[None, :]) / tau) - torch.sigmoid(
        (py[:, None] - (gy + bh * 0.5)[None, :]) / tau
    )
    density = sy.T @ sx
    flat = density.reshape(-1)
    k = max(1, int(flat.shape[0] * top_frac))
    top, _ = torch.topk(flat, k, sorted=False)
    return top.mean()


def _pairwise_overlap_loss(pos_h, sizes_h, avg_macro_area):
    """Sum of squared (overlap_area / avg_macro_area) over hard-macro pairs."""
    nh = pos_h.shape[0]
    if nh <= 1:
        return pos_h.sum() * 0.0
    dx = pos_h[:, 0:1] - pos_h[:, 0:1].T
    dy = pos_h[:, 1:2] - pos_h[:, 1:2].T
    sep_x = (sizes_h[:, 0:1] + sizes_h[:, 0:1].T) * 0.5
    sep_y = (sizes_h[:, 1:2] + sizes_h[:, 1:2].T) * 0.5
    over = torch.relu(sep_x - dx.abs()) * torch.relu(sep_y - dy.abs()) / max(avg_macro_area, 1e-12)
    over = over.masked_fill(torch.eye(nh, dtype=torch.bool, device=over.device), 0.0)
    return 0.5 * (over * over).sum()


class AnalyticalPlacer(_V2Placer):
    """v3 = v1 base + joint hard+soft polish (no hard-freeze) + legalize."""

    def __init__(
        self,
        *args,
        joint_polish_iters: int = 120,
        joint_polish_lr_scale: float = 0.05,
        joint_polish_dw_scale: float = 0.6,
        joint_polish_cw_scale: float = 1.0,
        joint_overlap_weight: float = 1.0,
        joint_pin_density_weight: float = 0.05,
        gate_with_real_proxy: bool = True,
        joint_legalize_gap: float = 0.05,
        run_v2_soft_polish_after: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.joint_polish_iters = joint_polish_iters
        self.joint_polish_lr_scale = joint_polish_lr_scale
        self.joint_polish_dw_scale = joint_polish_dw_scale
        self.joint_polish_cw_scale = joint_polish_cw_scale
        self.joint_overlap_weight = joint_overlap_weight
        self.joint_pin_density_weight = joint_pin_density_weight
        self.gate_with_real_proxy = gate_with_real_proxy
        self.joint_legalize_gap = joint_legalize_gap
        self.run_v2_soft_polish_after = run_v2_soft_polish_after

    def _real_proxy_score(self, placement: torch.Tensor, benchmark) -> float:
        """Return TILOS proxy cost for `placement`, or +inf if unavailable."""
        if not self.gate_with_real_proxy or _compute_proxy_cost is None:
            return float("inf")
        plc = getattr(benchmark, "_plc", None)
        if plc is None:
            return float("inf")
        try:
            costs = _compute_proxy_cost(placement.detach().cpu().float(), benchmark, plc)
            return float(costs["proxy_cost"])
        except Exception:
            return float("inf")

    def _joint_polish(self, placement: torch.Tensor, benchmark) -> torch.Tensor:
        nh = benchmark.num_hard_macros
        nm = benchmark.num_macros
        if self.joint_polish_iters <= 0:
            return placement

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dt = torch.float32 if dev.type == "cuda" else torch.float64
        placement = placement.to(dev, dt)

        cw_can = float(benchmark.canvas_width)
        ch_can = float(benchmark.canvas_height)
        diag = math.hypot(cw_can, ch_can)
        gr, gc = benchmark.grid_rows, benchmark.grid_cols
        bw, bh = cw_can / gc, ch_can / gr

        sizes = benchmark.macro_sizes.to(dev, dt)
        fixed = benchmark.macro_fixed.to(dev)

        net_idx, net_mask, port_pos = _tp._build_nets(benchmark, dev, dt)
        if net_idx.shape[0] == 0:
            return placement

        gx = (torch.arange(gc, device=dev, dtype=dt) + 0.5) * bw
        gy = (torch.arange(gr, device=dev, dtype=dt) + 0.5) * bh
        gamma = max(diag * self.gamma_e * 1.5, 0.1)

        pin_offsets, pin_owner = _build_pin_world_index(
            getattr(benchmark, "macro_pin_offsets", None) or [], nh, dev, dt,
        )

        with torch.no_grad():
            apos0 = torch.cat([placement, port_pos], dim=0)
            wl_0 = max(abs(_tp._wa_wirelength(apos0, net_idx, net_mask, gamma).item()), 1.0)
            den_0 = max(abs(_tp._density_topk(placement, sizes, nm, gx, gy, bw, bh).item()), 1e-6)
            cl_0 = max(abs(_tp._congestion_loss(apos0, net_idx, net_mask, gx, gy, bw, bh, gamma).item()), 1e-6)
            if pin_offsets is not None:
                pd_0 = max(
                    abs(_pin_density_topk(placement[:nh], pin_offsets, pin_owner, gx, gy, bw, bh).item()),
                    1e-6,
                )
            else:
                pd_0 = 1.0
            avg_macro_area = max(
                float((sizes[:nh, 0] * sizes[:nh, 1]).mean().item()) if nh > 0 else 1.0,
                1e-6,
            )

        dw = self.dw_p3 * float(self.joint_polish_dw_scale)
        tier = _tp.benchmark_stress_tier(benchmark) if self.adaptive_hard else 0
        cw_joint = _tp2._TIER_CW[tier] * 0.5 * float(self.joint_polish_cw_scale)
        over_w = float(self.joint_overlap_weight)
        pd_w = float(self.joint_pin_density_weight)

        hw, hh = sizes[:, 0] / 2, sizes[:, 1] / 2
        lb = torch.stack([hw, hh], dim=1)
        ub = torch.stack([cw_can - hw, ch_can - hh], dim=1)

        def score(x: torch.Tensor) -> float:
            with torch.no_grad():
                apos = torch.cat([x, port_pos], dim=0)
                wl = (_tp._wa_wirelength(apos, net_idx, net_mask, gamma) / wl_0).item()
                den = (_tp._density_topk(x, sizes, nm, gx, gy, bw, bh) / den_0).item()
                cl = (_tp._congestion_loss(apos, net_idx, net_mask, gx, gy, bw, bh, gamma) / cl_0).item()
                return float(wl + dw * den + cw_joint * cl)

        p = placement.clone().requires_grad_(True)
        lr = max(1e-4, self.lr * float(self.joint_polish_lr_scale))
        opt = torch.optim.Adam([p], lr=lr)

        for _ in range(self.joint_polish_iters):
            opt.zero_grad()
            apos = torch.cat([p, port_pos], dim=0)
            loss = (
                _tp._wa_wirelength(apos, net_idx, net_mask, gamma) / wl_0
                + dw * _tp._density_topk(p, sizes, nm, gx, gy, bw, bh) / den_0
                + cw_joint * _tp._congestion_loss(apos, net_idx, net_mask, gx, gy, bw, bh, gamma) / cl_0
            )
            if pd_w > 0.0 and pin_offsets is not None and nh >= 1:
                loss = loss + pd_w * _pin_density_topk(p[:nh], pin_offsets, pin_owner, gx, gy, bw, bh) / pd_0
            if over_w > 0.0 and nh >= 2:
                loss = loss + over_w * _pairwise_overlap_loss(p[:nh], sizes[:nh], avg_macro_area)

            loss.backward()
            with torch.no_grad():
                p.grad[fixed] = 0.0
                torch.nn.utils.clip_grad_norm_([p], max_norm=diag * 0.5)
            opt.step()
            with torch.no_grad():
                p.data = torch.max(torch.min(p.data, ub), lb)
                p.data[fixed] = placement[fixed]

        end_p = p.detach()

        try:
            legal_p = _tp._legalize(
                end_p, sizes, fixed, nh, cw_can, ch_can, gap=float(self.joint_legalize_gap)
            )
        except Exception:
            legal_p = end_p

        base_internal = score(placement)
        end_internal = score(end_p)
        legal_internal = score(legal_p)
        base_real = self._real_proxy_score(placement, benchmark)
        legal_real = self._real_proxy_score(legal_p, benchmark)

        use_real = (
            self.gate_with_real_proxy
            and base_real != float("inf")
            and legal_real != float("inf")
        )
        if use_real:
            improved = legal_real < base_real - 1e-5
        else:
            improved = legal_internal < base_internal - 1e-5
        result_p = legal_p if improved else placement
        result_name = "legal" if improved else "base"

        if self.verbose:
            gate_tag = "real" if use_real else "internal"
            real_tag = (
                f" real_base={base_real:.4f} real_legal={legal_real:.4f}" if use_real else ""
            )
            print(
                f"  [{benchmark.name}] Joint polish: tier={tier} "
                f"base={base_internal:.4f} -> end={end_internal:.4f} "
                f"legal={legal_internal:.4f}{real_tag} "
                f"(gate={gate_tag} kept={result_name})",
                flush=True,
            )
        return result_p

    def place(self, benchmark):
        base = _BasePlacer.place(self, benchmark)
        polished = self._joint_polish(base, benchmark)

        if self.run_v2_soft_polish_after:
            pre_soft = polished
            soft = self._soft_only_polish(pre_soft, benchmark)
            pre_real = self._real_proxy_score(pre_soft, benchmark)
            soft_real = self._real_proxy_score(soft, benchmark)
            use_real = (
                self.gate_with_real_proxy
                and pre_real != float("inf")
                and soft_real != float("inf")
            )
            if use_real:
                kept_soft = soft_real < pre_real - 1e-5
                polished = soft if kept_soft else pre_soft
                if self.verbose:
                    print(
                        f"  [{benchmark.name}] Soft gate: "
                        f"pre={pre_real:.4f} -> soft={soft_real:.4f} "
                        f"(gate=real kept={'soft' if kept_soft else 'pre_soft'})",
                        flush=True,
                    )
            else:
                polished = soft

        if isinstance(polished, torch.Tensor):
            return polished.cpu().float()
        return polished
