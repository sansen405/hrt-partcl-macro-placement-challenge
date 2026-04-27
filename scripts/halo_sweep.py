"""Quick stress test: how much does halo_frac alone change the proxy?

Runs ``AnalyticalPlacer`` from ``submissions/will_seed/tierplace.py`` on a
small set of representative benchmarks for each halo in HALOS, and prints
the resulting proxy cost spread.

Goal: decide whether trivial knob variation already produces meaningful
basin diversity (±0.05 proxy or more) before building a full multi-start
ensemble.

Usage:
    uv run python scripts/halo_sweep.py
"""

from __future__ import annotations

import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "submissions" / "will_seed"
sys.path.insert(0, str(SUB))

import tierplace as tp  # noqa: E402  (must come after sys.path edit)
from macro_place.evaluate import evaluate_benchmark  # noqa: E402

HALOS = [0.04, 0.06, 0.08, 0.10, 0.12]
BENCHES = ["ibm01", "ibm08", "ibm10", "ibm17"]
TESTCASE_ROOT = str(ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04")


def main() -> None:
    print(
        f"{'bench':>8} {'halo':>6} {'proxy':>9} {'wl':>7} {'den':>7} "
        f"{'cong':>7} {'ovl':>4} {'time':>7}",
        flush=True,
    )
    print("-" * 60, flush=True)

    results: dict[tuple[str, float], dict] = {}
    t0 = time.time()
    for bench in BENCHES:
        for halo in HALOS:
            placer = tp.AnalyticalPlacer(halo_frac=halo, verbose=False)
            buf = io.StringIO()
            try:
                with redirect_stdout(buf):
                    r = evaluate_benchmark(placer, bench, TESTCASE_ROOT)
            except Exception as exc:  # noqa: BLE001
                print(f"{bench:>8} {halo:>6.2f}  ERROR: {exc}", flush=True)
                continue
            results[(bench, halo)] = r
            print(
                f"{bench:>8} {halo:>6.2f} {r['proxy_cost']:>9.4f} "
                f"{r['wirelength']:>7.3f} {r['density']:>7.3f} "
                f"{r['congestion']:>7.3f} {r['overlaps']:>4d} "
                f"{r['runtime']:>6.1f}s",
                flush=True,
            )

    print(flush=True)
    print(f"{'bench':>8} {'min':>9} {'max':>9} {'spread':>9} {'mean':>9}", flush=True)
    print("-" * 50, flush=True)
    for bench in BENCHES:
        proxys = [
            results[(bench, h)]["proxy_cost"]
            for h in HALOS
            if (bench, h) in results
        ]
        if not proxys:
            print(f"{bench:>8}  no results", flush=True)
            continue
        pmin, pmax = min(proxys), max(proxys)
        pmean = sum(proxys) / len(proxys)
        print(
            f"{bench:>8} {pmin:>9.4f} {pmax:>9.4f} {pmax - pmin:>9.4f} "
            f"{pmean:>9.4f}",
            flush=True,
        )

    print(flush=True)
    print(f"total wall time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
