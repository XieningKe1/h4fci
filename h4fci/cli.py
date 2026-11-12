"""Command-line entry point of the H4+ FCI program.

Nothing here imports pyscf.  The Gaussian integrals (``h4fci/integrals.py``),
the ROHF/UHF references (``h4fci/scf.py``) and the determinant FCI
(``h4fci/fci.py``) are all implemented in this package.  The file
``h4fci/data/reference_pyscf.json`` only holds pyscf's numbers so the program
can print a comparison column.

Typical use::

    python -m h4fci --basis cc-pVTZ --threads 20 --nroots 3
    h4-fci --basis cc-pVDZ                    # after `pip install -e .`
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from .basis import Molecule
from .fci import FCI, transform_integrals
from .integrals import IntegralEngine
from .scf import ROHF, UHF

HARTREE2EV = 27.211386245988
REF_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "reference_pyscf.json")


def square_h4(side=2.0):
    h = side / 2.0
    return [("H", (h, h, 0.0)), ("H", (h, -h, 0.0)),
            ("H", (-h, -h, 0.0)), ("H", (-h, h, 0.0))]


def run_scf(cls, mol, S, h, eri, **kw):
    """SCF with a damped-retry fallback for nearly degenerate open shells."""
    scf = cls(mol, S, h, eri, max_cycle=300, conv_tol=1e-10, diis=True, **kw)
    E = scf.run()
    if not scf.converged:
        scf = cls(mol, S, h, eri, max_cycle=900, conv_tol=1e-8,
                  diis=False, damping=0.1, **kw)
        E = scf.run()
    return scf, E


def solve_fci(ci, nroots, tol=1e-9):
    from scipy.sparse.linalg import LinearOperator, eigsh
    calls = [0]

    def mv(x):
        calls[0] += 1
        return ci.sigma(x.reshape(ci.Na, ci.Nb)).ravel()

    op = LinearOperator((ci.n_det, ci.n_det), matvec=mv, dtype=float)
    w, V = eigsh(op, k=nroots, which="SA", tol=tol, maxiter=600, ncv=24)
    order = np.argsort(w)
    w = w[order]
    ci.vectors = [V[:, k].reshape(ci.Na, ci.Nb) for k in order]
    ci.energies = w
    return w, calls[0]


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="h4-fci",
        description="Self-contained Hartree-Fock + full-CI for H4+")
    ap.add_argument("--basis", default="cc-pVTZ")
    ap.add_argument("--side", type=float, default=2.0)
    ap.add_argument("--nroots", type=int, default=3)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if args.threads:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[v] = str(args.threads)

    t_start = time.time()
    atoms = square_h4(args.side)
    mol = Molecule(atoms, args.basis)
    print(f"H4+  basis={args.basis}  nao={mol.nao}  shells={len(mol.shells)}  "
          f"Enuc={mol.enuc:.12f}")
    print(f"geometry (bohr): {np.round(mol.coords, 4).tolist()}")

    # ---------------- integrals (own engine, cached) ------------------
    os.makedirs(args.cache, exist_ok=True)
    fcache = os.path.join(args.cache, f"h4_{args.basis}_{args.side:.2f}.npz")
    t0 = time.time()
    if os.path.exists(fcache):
        d = np.load(fcache)
        S, T, V, eri = d["S"], d["T"], d["V"], d["eri"]
        print(f"integrals: loaded from {fcache} ({time.time()-t0:.1f}s)")
    else:
        eng = IntegralEngine(mol)
        S, T, V = eng.one_electron()
        eri = eng.eri(verbose=True)
        np.savez(fcache, S=S, T=T, V=V, eri=eri)
        print(f"integrals: computed ({time.time()-t0:.1f}s)")
    t_int = time.time() - t_start
    h = T + V
    print(f"eri tensor {eri.shape} = {eri.nbytes/1e6:.0f} MB")

    # ---------------- SCF references ----------------------------------
    t0 = time.time()
    rohf, e_rohf = run_scf(ROHF, mol, S, h, eri, nocc_c=1, nocc_o=1)
    print(f"\nROHF: E = {e_rohf:.12f}  conv={rohf.converged}  "
          f"cycles={rohf.ncycles}  ({time.time()-t0:.1f}s)")
    t0 = time.time()
    uhf, e_uhf = run_scf(UHF, mol, S, h, eri, nalpha=2, nbeta=1)
    print(f"UHF : E = {e_uhf:.12f}  <S^2> = {uhf.s2():.6f}  "
          f"conv={uhf.converged}  cycles={uhf.ncycles}  ({time.time()-t0:.1f}s)")

    # ---------------- FCI ---------------------------------------------
    t0 = time.time()
    h_mo, eri_mo = transform_integrals(h, eri, rohf.C)
    ci = FCI(h_mo, eri_mo, nalpha=2, nbeta=1)
    print(f"\nFCI setup ({time.time()-t0:.1f}s): norb={mol.nao}, "
          f"determinants={ci.n_det} (= C({mol.nao},2)*C({mol.nao},1))")
    t0 = time.time()
    e_fci, nmatvec = solve_fci(ci, args.nroots)
    t_fci = time.time() - t0
    print(f"FCI diagonalisation: {t_fci:.1f}s ({nmatvec} sigma evaluations)")

    # ---------------- comparison column (pyscf numbers, optional) -----
    ref = None
    if os.path.exists(REF_FILE):
        try:
            raw = json.load(open(REF_FILE))
            ref = raw["bases"].get(args.basis.upper())
        except Exception:
            ref = None

    print("\n" + "=" * 74)
    print("summary (Hartree, total energies with nuclear repulsion)")
    print("=" * 74)
    print(f"  E(ROHF) = {e_rohf:.12f}")
    print(f"  E(UHF)  = {e_uhf:.12f}   <S^2> = {uhf.s2():.6f}")
    for k in range(len(e_fci)):
        line = f"  E(FCI,{k}) = {e_fci[k] + mol.enuc:.12f}"
        if ref is not None and k < len(ref["E_fci"]):
            d = e_fci[k] + mol.enuc - float(ref["E_fci"][k])
            line += f"   (pyscf {float(ref['E_fci'][k]):.12f}, diff {d:+.1e})"
        print(line)
    corr = e_fci[0] + mol.enuc - e_rohf
    print(f"\n  correlation energy (FCI0 - ROHF) = {corr:+.12f} Ha "
          f"= {corr*HARTREE2EV:+.4f} eV")
    if ref is not None:
        print(f"\n  vs pyscf reference:  ROHF {e_rohf - ref['E_rohf']:+.1e} Ha, "
              f"UHF {e_uhf - ref['E_uhf']:+.1e} Ha")
        print(f"  <S^2> ours {uhf.s2():.6f} vs pyscf {ref['s2_uhf']:.6f}")

    if args.json:
        out = {"basis": args.basis, "side_bohr": args.side, "nao": mol.nao,
               "ndet": ci.n_det, "E_nuc": mol.enuc, "E_rohf": e_rohf,
               "E_uhf": e_uhf, "s2_uhf": uhf.s2(),
               "E_fci": [float(x + mol.enuc) for x in e_fci],
               "corr_energy": float(corr),
               "timings": {"integrals_s": t_int, "fci_s": t_fci,
                           "sigma_calls": nmatvec}}
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nresults written to {args.json}")
    print(f"\ntotal wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":   # pragma: no cover
    main()
