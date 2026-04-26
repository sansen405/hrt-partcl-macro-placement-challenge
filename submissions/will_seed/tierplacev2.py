"""
TierPlace v2

TierPlace baseline + a final soft-macro polish that runs on GPU when
available. Two methods are exposed:

* ``polish_method="adam"`` (default, proven): a gradient pass on the same
  WL + dw·density + cw·congestion proxy used by the base placer. With
  dw≫1 this is effectively a density-spread step over soft macros.
* ``polish_method="centroid"``: gradient-free Jacobi sweep on the L2
  clique-1/(k-1) WL model. Cheap and reduces WL substantially, but tends
  to pile soft macros on top of each other (density blows up), so it
  almost always reverts via the score gate. Kept here for experimentation.

Both methods share the same "no-worse" safeguard: the polished placement
is only kept if its (WL + dw·den + cw·cl) score is strictly better than
the input.
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

_TIER_CW = (0.05, 0.15, 0.25)


class AnalyticalPlacer(_BaseTierPlace):
    def __init__(
        self,
        *args,
        polish_method: str = "adam",
        soft_polish_iters: int = 80,
        soft_polish_lr_scale: float = 0.15,
        soft_polish_dw_scale: float = 0.8,
        soft_polish_cw_scale: float = 1.2,
        soft_polish_damping: float = 0.5,
        soft_polish_score_every: int = 5,
        soft_polish_patience: int = 3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if polish_method not in ("adam", "centroid"):
            raise ValueError(f"polish_method must be 'adam' or 'centroid', got {polish_method!r}")
        self.polish_method = polish_method
        self.soft_polish_iters = soft_polish_iters
        self.soft_polish_lr_scale = soft_polish_lr_scale
        self.soft_polish_dw_scale = soft_polish_dw_scale
        self.soft_polish_cw_scale = soft_polish_cw_scale
        self.soft_polish_damping = soft_polish_damping
        self.soft_polish_score_every = soft_polish_score_every
        self.soft_polish_patience = soft_polish_patience

    def _polish_setup(self, placement: torch.Tensor, benchmark):
        """Common scaffolding for both polish methods.

        Returns a context dict, or ``None`` to skip polishing entirely.
        """
        nh = benchmark.num_hard_macros
        nm = benchmark.num_macros
        if nm <= nh or self.soft_polish_iters <= 0:
            return None

        # super().place returns CPU float32; move onto GPU when available so
        # the polish actually uses the accelerator.
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dt = torch.float32 if dev.type == "cuda" else torch.float64
        placement = placement.to(dev, dt)

        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        diag = math.hypot(cw, ch)
        gr, gc = benchmark.grid_rows, benchmark.grid_cols
        bw, bh = cw / gc, ch / gr

        sizes = benchmark.macro_sizes.to(dev, dt)
        fixed = benchmark.macro_fixed.to(dev)
        soft_movable = torch.zeros(nm, dtype=torch.bool, device=dev)
        soft_movable[nh:] = True
        soft_movable &= ~fixed
        if not soft_movable.any():
            return None

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

        dw = self.dw_p3 * self.soft_polish_dw_scale
        tier = _tp.benchmark_stress_tier(benchmark) if self.adaptive_hard else 0
        cw_polish = _TIER_CW[tier] * 0.5 * self.soft_polish_cw_scale

        hw = sizes[:, 0] / 2
        hh = sizes[:, 1] / 2
        return {
            "dev": dev, "dt": dt,
            "nh": nh, "nm": nm,
            "cw": cw, "ch": ch, "diag": diag, "bw": bw, "bh": bh,
            "sizes": sizes, "fixed": fixed, "soft_movable": soft_movable,
            "net_idx": net_idx, "net_mask": net_mask, "port_pos": port_pos,
            "gx": gx, "gy": gy, "gamma": gamma,
            "wl_0": wl_0, "den_0": den_0, "cl_0": cl_0,
            "dw": dw, "cw_polish": cw_polish,
            "hw": hw, "hh": hh,
            "lb": torch.stack([hw, hh], dim=1),
            "ub": torch.stack([cw - hw, ch - hh], dim=1),
            "placement": placement,
        }

    def _score_fn(self, ctx):
        net_idx, net_mask = ctx["net_idx"], ctx["net_mask"]
        port_pos = ctx["port_pos"]
        sizes = ctx["sizes"]
        nm = ctx["nm"]
        gx, gy, bw, bh = ctx["gx"], ctx["gy"], ctx["bw"], ctx["bh"]
        gamma = ctx["gamma"]
        wl_0, den_0, cl_0 = ctx["wl_0"], ctx["den_0"], ctx["cl_0"]
        dw, cw_polish = ctx["dw"], ctx["cw_polish"]

        def _score(x: torch.Tensor) -> float:
            with torch.no_grad():
                apos = torch.cat([x, port_pos], dim=0)
                wl = float((_tp._wa_wirelength(apos, net_idx, net_mask, gamma) / wl_0).item())
                den = float((_tp._density_topk(x, sizes, nm, gx, gy, bw, bh) / den_0).item())
                cl = float((_tp._congestion_loss(apos, net_idx, net_mask, gx, gy, bw, bh, gamma) / cl_0).item())
                return wl + dw * den + cw_polish * cl

        return _score

    def _soft_only_polish_adam(self, ctx, benchmark):
        """Gradient pass on (WL + dw·density + cw·congestion). Density-driven
        when dw is large, which is the default."""
        dev, dt = ctx["dev"], ctx["dt"]
        nh, nm = ctx["nh"], ctx["nm"]
        sizes, fixed = ctx["sizes"], ctx["fixed"]
        net_idx, net_mask, port_pos = ctx["net_idx"], ctx["net_mask"], ctx["port_pos"]
        gx, gy, bw, bh = ctx["gx"], ctx["gy"], ctx["bw"], ctx["bh"]
        gamma = ctx["gamma"]
        wl_0, den_0, cl_0 = ctx["wl_0"], ctx["den_0"], ctx["cl_0"]
        dw, cw_polish = ctx["dw"], ctx["cw_polish"]
        lb, ub = ctx["lb"], ctx["ub"]
        placement = ctx["placement"]

        p = placement.clone().requires_grad_(True)
        lr = max(1e-4, self.lr * self.soft_polish_lr_scale)
        opt = torch.optim.Adam([p], lr=lr)

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
        return p.detach()

    def _soft_only_polish_centroid(self, ctx, benchmark):
        """Jacobi sweep on the L2 clique-1/(k-1) WL model.

        Each iteration moves every soft macro toward the average across its
        nets of the centroid of "other pins" in that net. Gradient-free,
        cheap, and provably reduces clique-WL — but ignores density and
        congestion, so the score gate often reverts on this proxy.
        """
        dev, dt = ctx["dev"], ctx["dt"]
        nh, nm = ctx["nh"], ctx["nm"]
        cw, ch = ctx["cw"], ctx["ch"]
        fixed, soft_movable = ctx["fixed"], ctx["soft_movable"]
        net_idx, net_mask, port_pos = ctx["net_idx"], ctx["net_mask"], ctx["port_pos"]
        hw, hh = ctx["hw"], ctx["hh"]
        placement = ctx["placement"]

        n_total = nm + port_pos.shape[0]
        # _build_nets only keeps nets with ≥2 pins, so (k-1) ≥ 1 always.
        pin_count = net_mask.sum(1).to(dt)
        inv_kminus1 = (1.0 / (pin_count - 1.0).clamp(min=1.0)).view(-1, 1, 1)

        flat_idx = net_idx.reshape(-1)
        flat_msk = net_mask.reshape(-1).to(dt)
        idx_expand = flat_idx.unsqueeze(-1).expand(-1, 2)
        msk_f = net_mask.unsqueeze(-1).to(dt)

        # node_cnt is invariant across iterations.
        node_cnt = torch.zeros(n_total, device=dev, dtype=dt)
        node_cnt.scatter_add_(0, flat_idx, flat_msk)
        node_cnt_inv = 1.0 / node_cnt.clamp(min=1.0)
        node_sum = torch.zeros(n_total, 2, device=dev, dtype=dt)

        score = self._score_fn(ctx)
        base_score = score(placement)
        best_p = placement.clone()
        best_score = base_score

        damping = float(self.soft_polish_damping)
        check_every = max(1, int(self.soft_polish_score_every))
        patience = max(1, int(self.soft_polish_patience))

        p = placement.clone()
        plateau = 0
        ran_iters = 0
        for it in range(self.soft_polish_iters):
            all_pos = torch.cat([p, port_pos], dim=0)
            pin_pos = all_pos[net_idx] * msk_f                # [n_nets, k_max, 2]
            net_sum = pin_pos.sum(dim=1, keepdim=True)        # [n_nets, 1, 2]
            contrib = (net_sum - pin_pos) * inv_kminus1
            contrib = contrib * msk_f

            node_sum.zero_()
            node_sum.scatter_add_(0, idx_expand, contrib.reshape(-1, 2))
            node_avg = node_sum * node_cnt_inv.unsqueeze(-1)

            target = node_avg[:nm]
            p_new = p.clone()
            p_new[soft_movable] = (
                (1.0 - damping) * p[soft_movable] + damping * target[soft_movable]
            )
            p_new[:, 0] = torch.minimum(torch.maximum(p_new[:, 0], hw), cw - hw)
            p_new[:, 1] = torch.minimum(torch.maximum(p_new[:, 1], hh), ch - hh)
            p_new[:nh] = placement[:nh]
            p_new[fixed] = placement[fixed]

            ran_iters = it + 1
            is_check = (it + 1) % check_every == 0 or it == self.soft_polish_iters - 1
            if is_check:
                s = score(p_new)
                if s < best_score - 1e-5:
                    best_score = s
                    best_p = p_new.clone()
                    plateau = 0
                else:
                    plateau += 1
                    if plateau >= patience:
                        p = p_new
                        break
            p = p_new

        if self.verbose:
            kept = best_score < base_score
            print(
                f"  [{benchmark.name}] Soft centroid: "
                f"iters={ran_iters} base={base_score:.4f} -> best={best_score:.4f} "
                f"({'kept' if kept else 'reverted'})",
                flush=True,
            )
        return best_p, best_score, base_score

    def _soft_only_polish(self, placement: torch.Tensor, benchmark):
        ctx = self._polish_setup(placement, benchmark)
        if ctx is None:
            return placement

        score = self._score_fn(ctx)
        base_score = score(ctx["placement"])

        if self.polish_method == "centroid":
            polished, polished_score, _ = self._soft_only_polish_centroid(ctx, benchmark)
        else:
            polished = self._soft_only_polish_adam(ctx, benchmark)
            polished_score = score(polished)
            if self.verbose:
                kept = polished_score <= base_score
                print(
                    f"  [{benchmark.name}] Soft Adam: "
                    f"base={base_score:.4f} -> polished={polished_score:.4f} "
                    f"({'kept' if kept else 'reverted'})",
                    flush=True,
                )

        keep = polished_score <= base_score
        result = polished if keep else ctx["placement"]
        return result.cpu().float()

    def place(self, benchmark):
        base = super().place(benchmark)
        return self._soft_only_polish(base, benchmark)
