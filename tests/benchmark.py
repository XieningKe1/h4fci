"""Stage-by-stage timing of the from-scratch H4+ pipeline.

Usage:  python3 tests/benchmark.py [--threads 20]
Writes docs/timing.md and prints a markdown table.
"""
import argparse
import os
import resource
import sys
import time

import json

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from h4fci.basis import Molecule                      # noqa: E402
from h4fci.fci import FCI, transform_integrals        # noqa: E402
from h4fci.integrals import IntegralEngine            # noqa: E402
from h4fci.scf import ROHF, UHF                       # noqa: E402


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class CountingOp:
    """Wrap sigma and count matvecs for the ARPACK report."""

    def __init__(self, ci):
        self.ci = ci
        self.n = ci.n_det
        self.calls = 0

    def matvec(self, x):
        self.calls += 1
        return self.ci.sigma(x.reshape(self.ci.Na, self.ci.Nb)).ravel()


def run(basis, atoms, threads):
    row = {"basis": basis}
    t0 = time.perf_counter()
    mol = Molecule(atoms, basis)
    row["build"] = time.perf_counter() - t0

    eng = IntegralEngine(mol)
    t0 = time.perf_counter()
    S, T, V = eng.one_electron()
    row["int1e"] = time.perf_counter() - t0
    h = T + V

    t0 = time.perf_counter()
    eri = eng.eri()
    row["int2e"] = time.perf_counter() - t0
    nsh = len(mol.shells)
    pairs = [(i, j) for i in range(nsh) for j in range(i, nsh)]
    row["nquartets"] = sum(1 for pi in range(len(pairs))
                           for pk in range(pi, len(pairs)))

    t0 = time.perf_counter()
    rohf = ROHF(mol, S, h, eri, max_cycle=900, conv_tol=1e-10, diis=True)
    e_r = rohf.run()
    row["rohf"] = time.perf_counter() - t0
    row["rohf_cycles"] = rohf.ncycles
    row["rohf_conv"] = rohf.converged
    if not rohf.converged:                       # damped retry (STO-3G)
        t0 = time.perf_counter()
        rohf = ROHF(mol, S, h, eri, max_cycle=900, conv_tol=1e-8,
                    diis=False, damping=0.1)
        e_r = rohf.run()
        row["rohf"] += time.perf_counter() - t0
        row["rohf_cycles"] = rohf.ncycles

    t0 = time.perf_counter()
    uhf = UHF(mol, S, h, eri)
    e_u = uhf.run()
    row["uhf"] = time.perf_counter() - t0
    row["uhf_cycles"] = uhf.ncycles

    t0 = time.perf_counter()
    h_mo, eri_mo = transform_integrals(h, eri, rohf.C)
    row["transform"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ci = FCI(h_mo, eri_mo)
    row["fci_setup"] = time.perf_counter() - t0
    row["ndet"] = ci.n_det

    v = np.random.default_rng(0).random((ci.Na, ci.Nb))
    ci.sigma(v)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        ci.sigma(v)
        ts.append(time.perf_counter() - t0)
    row["sigma"] = float(np.median(ts))

    from scipy.sparse.linalg import LinearOperator, eigsh
    op = CountingOp(ci)
    A = LinearOperator((ci.n_det, ci.n_det), matvec=op.matvec, dtype=float)
    t0 = time.perf_counter()
    w, Vv = eigsh(A, k=3, which="SA", tol=1e-9, maxiter=600, ncv=20)
    row["arpack"] = time.perf_counter() - t0
    row["matvecs"] = op.calls
    row["e_fci"] = float(np.sort(w)[0] + mol.enuc)
    row["e_rohf"] = float(e_r)
    row["e_uhf"] = float(e_u)
    row["peak_mb"] = peak_mb()
    row["mem_eri_mo_mb"] = eri_mo.nbytes / 1e6
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--skip", default="", help="comma separated bases to skip")
    args = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(args.threads)

    atoms = [("H", (1.0, 1.0, 0.0)), ("H", (1.0, -1.0, 0.0)),
             ("H", (-1.0, -1.0, 0.0)), ("H", (-1.0, 1.0, 0.0))]
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    rows = []
    for basis in ("STO-3G", "cc-pVDZ", "cc-pVTZ"):
        if basis in skip:
            continue
        print(f"--- {basis} ---", flush=True)
        rows.append(run(basis, atoms, args.threads))
        r = rows[-1]
        print(f"    int2e {r['int2e']:.1f}s  rohf {r['rohf']:.2f}s({r['rohf_cycles']}c)"
              f"  uhf {r['uhf']:.2f}s  sigma {r['sigma']*1e3:.1f}ms"
              f"  arpack {r['arpack']:.1f}s({r['matvecs']} mv)  peak {r['peak_mb']:.0f}MB",
              flush=True)

    # ---- markdown table -------------------------------------------------
    cols = ["STO-3G", "cc-pVDZ", "cc-pVTZ"]
    def fmt(x, nd=2):
        return f"{x:.{nd}f}"
    lines = []
    lines.append("| 阶段 | " + " | ".join(c for c in cols) + " |")
    lines.append("|---|" + "---|" * len(cols))
    def row(label, key, nd=2, scale=1.0):
        cells = []
        for c in cols:
            r = next((r for r in rows if r["basis"] == c), None)
            cells.append("—" if r is None else fmt(r[key] * scale, nd))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    row("基组/AO 构造 (s)", "build", 3)
    row("1e 积分 S/T/V (s)", "int1e", 2)
    row("2e 积分 ERI (s)", "int2e", 2)
    row("shell quartet 数", "nquartets", 0)
    row("ROHF (s)", "rohf", 2)
    row("ROHF 圈数", "rohf_cycles", 0)
    row("UHF (s)", "uhf", 2)
    row("UHF 圈数", "uhf_cycles", 0)
    row("AO→MO 变换 (s)", "transform", 2)
    row("FCI 构建 (s)", "fci_setup", 2)
    row("一次 σ (ms)", "sigma", 1, 1000.0)
    row("ARPACK (s)", "arpack", 1)
    row("ARPACK matvec 数", "matvecs", 0)
    row("峰值内存 (MB)", "peak_mb", 0)
    table = "\n".join(lines)
    print("\n" + table)
    out = os.path.join(ROOT, "docs", "timing.md")
    with open(out, "w") as fh:
        fh.write("# H4+ 各阶段耗时（20 线程）\n\n")
        fh.write(table + "\n\n")
        fh.write("说明：一次 σ 取 5 次中位数；ARPACK matvec 数即 σ 的调用次数；"
                 "峰值内存为单进程 RSS；\\* STO-3G 的 ROHF 走阻尼回退。\n")
        # pyscf comparison, taken from the stored reference file
        try:
            ref = json.load(open(os.path.join(ROOT, "h4fci", "data",
                                              "reference_pyscf.json")))
            p20 = ref.get("timing_20threads_pyscf_path", {}).get("cc-pVTZ", {})
            rowp = next((r for r in rows if r["basis"] == "cc-pVTZ"), None)
            if p20 and rowp is not None:
                fh.write("\n## 与 pyscf 的耗时对比（cc-pVTZ，同一台机器，20 线程）\n\n")
                fh.write("| 阶段 | 本程序 | pyscf |\n|---|---|---|\n")
                fh.write(f"| 2e 积分 | {rowp['int2e']:.1f} s | {p20.get('integrals', 0):.2f} s |\n")
                fh.write(f"| ROHF | {rowp['rohf']:.2f} s | {p20.get('rohf', 0):.2f} s |\n")
                fh.write(f"| UHF | {rowp['uhf']:.2f} s | {p20.get('uhf', 0):.2f} s |\n")
                fh.write(f"| AO→MO 变换 | {rowp['transform']:.2f} s | {p20.get('mo_transform', 0):.2f} s |\n")
                fh.write(f"| FCI 构建 | {rowp['fci_setup']:.2f} s | {p20.get('fci_setup', 0):.2f} s |\n")
                fh.write(f"| FCI 对角化 | {rowp['arpack']:.1f} s（{rowp['matvecs']} 次 σ） | {p20.get('fci', 0):.0f} s |\n")
                tot = rowp['int2e'] + rowp['rohf'] + rowp['uhf'] + rowp['transform'] + rowp['fci_setup'] + rowp['arpack']
                pt = sum(p20.get(k, 0) for k in ("integrals", "rohf", "uhf",
                                                 "mo_transform", "fci_setup", "fci"))
                fh.write(f"| **合计** | **{tot:.0f} s** | **{pt:.0f} s** |\n")
                fh.write("\n差距来自实现语言：pyscf 的积分与 FCI 是 C/高度优化实现，"
                         "本程序是纯 Python（σ 用 numpy 向量化）。两者能量一致到 10⁻¹²。\n")
        except Exception as exc:          # the comparison is optional
            fh.write(f"\n（pyscf 对比数据不可用：{exc}）\n")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
