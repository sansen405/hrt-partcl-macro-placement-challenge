"""
TierPlace v2

TierPlace baseline + extra soft-only final polish.
"""

import importlib.util
import math
from pathlib import Path

import torch


def _load_base_tierplace():
    path = Path(__file__).with_name("tierplace.py")
    spec = importlib.util.spec_from_file_location("tierplace_base_module", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load base TierPlace module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_tp = _load_base_tierplace()
_BaseTierPlace = _tp.AnalyticalPlacer


class AnalyticalPlacer(_BaseTierPlace):
    def __init__(
        self,
        soft_polish_iters: int = 80,
        soft_polish_lr_scale: float = 0.15,
        soft_polish_dw_scale: float = 0.8,
        soft_polish_cw_scale: float = 1.2,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.soft_polish_iters = soft_polish_iters
        self.soft_polish_lr_scale = soft_polish_lr_scale
        self.soft_polish_dw_scale = soft_polish_dw_scale
        self.soft_polish_cw_scale = soft_polish_cw_scale

    def _soft_only_polish(self, placement: torch.Tensor, benchmark):
        nh = benchmark.num_hard_macros
        nm = benchmark.num_macros
        if nm <= nh or self.soft_polish_iters <= 0:
            return placement

        dev = placement.device
        dt = placement.dtype
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        diag = math.hypot(cw, ch)
        gr, gc = benchmark.grid_rows, benchmark.grid_cols
        bw, bh = cw / gc, ch / gr

        sizes = benchmark.macro_sizes.to(dev, dt)
        fixed = benchmark.macro_fixed.to(dev)
        soft_mask = torch.zeros(nm, dtype=torch.bool, device=dev)
        soft_mask[nh:] = True
        movable_soft = soft_mask & (~fixed)
        if not movable_soft.any():
            return placement

        net_idx, net_mask, port_pos = _tp._build_nets(benchmark, dev, dt)
        if net_idx.shape[0] == 0:
            return placement

        gx = (torch.arange(gc, device=dev, dtype=dt) + 0.5) * bw
        gy = (torch.arange(gr, device=dev, dtype=dt) + 0.5) * bh
        gamma = max(diag * self.gamma_e * 1.5, 0.1)

        p = placement.clone().requires_grad_(True)
        lr = max(1e-4, self.lr * self.soft_polish_lr_scale)
        opt = torch.optim.Adam([p], lr=lr)

        hw = sizes[:, 0] / 2
        hh = sizes[:, 1] / 2
        lb = torch.stack([hw, hh], dim=1)
        ub = torch.stack([cw - hw, ch - hh], dim=1)

        with torch.no_grad():
            apos0 = torch.cat([p.detach(), port_pos], dim=0)
            wl_0 = max(abs(_tp._wa_wirelength(apos0, net_idx, net_mask, gamma).item()), 1.0)
            den_0 = max(abs(_tp._density_topk(p.detach(), sizes, nm, gx, gy, bw, bh).item()), 1e-6)
            cl_0 = max(abs(_tp._congestion_loss(apos0, net_idx, net_mask, gx, gy, bw, bh, gamma).item()), 1e-6)

        dw = self.dw_p3 * self.soft_polish_dw_scale
        tier = _tp.benchmark_stress_tier(benchmark) if self.adaptive_hard else 0
        cw_polish = [0.05, 0.15, 0.25][tier] * 0.5 * self.soft_polish_cw_scale

        def _score(x: torch.Tensor) -> float:
            with torch.no_grad():
                apos = torch.cat([x, port_pos], dim=0)
                wl = float((_tp._wa_wirelength(apos, net_idx, net_mask, gamma) / wl_0).item())
                den = float((_tp._density_topk(x, sizes, nm, gx, gy, bw, bh) / den_0).item())
                cl = float((_tp._congestion_loss(apos, net_idx, net_mask, gx, gy, bw, bh, gamma) / cl_0).item())
                return wl + dw * den + cw_polish * cl

        base_score = _score(placement)

        for _ in range(self.soft_polish_iters):
            opt.zero_grad()
            apos = torch.cat([p, port_pos], dim=0)
            wl = _tp._wa_wirelength(apos, net_idx, net_mask, gamma) / wl_0
            den = _tp._density_topk(p, sizes, nm, gx, gy, bw, bh) / den_0
            cl = _tp._congestion_loss(apos, net_idx, net_mask, gx, gy, bw, bh, gamma) / cl_0
            loss = wl + dw * den + cw_polish * cl
            loss.backward()
            with torch.no_grad():
                p.grad[:nh] = 0.0
                p.grad[fixed] = 0.0
            opt.step()
            with torch.no_grad():
                p.data = torch.max(torch.min(p.data, ub), lb)
                p.data[:nh] = placement[:nh]
                p.data[fixed] = placement[fixed]

        polished = p.detach()
        # Hard upper bound: never return worse-than-base polished result.
        return polished if _score(polished) <= base_score else placement

    def place(self, benchmark):
        base = super().place(benchmark)
        return self._soft_only_polish(base, benchmark)
