"""Independent validation of the self-written integral engine.

No external quantum-chemistry package is involved: s-type integrals have
closed forms, and every p/d integral is checked against finite differences of
those closed forms (multiplying a gaussian by (x-A) is a derivative with
respect to the centre, which tests the Hermite recursion and all sign
conventions).  A H2/STO-3G SCF energy is checked against the literature value
as an end-to-end test.

    python3 tests/test_integrals.py
"""
import sys
import os
import numpy as np
from scipy.special import erf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from h4fci.basis import Shell, cart2sph_matrix, Molecule      # noqa: E402
from h4fci.integrals import IntegralEngine                    # noqa: E402
from h4fci.scf import ROHF                                    # noqa: E402


def F0(T):
    T = np.asarray(T, float)
    out = np.ones_like(T)
    m = T > 1e-14
    out[m] = 0.5 * np.sqrt(np.pi / T[m]) * erf(np.sqrt(T[m]))
    return out


def s_ov(a, A, b, B):
    p = a + b
    return (np.pi / p) ** 1.5 * np.exp(-a * b / p * np.sum((A - B) ** 2))


def s_kin(a, A, b, B):
    p = a + b
    mu = a * b / p
    return mu * (3.0 - 2.0 * mu * np.sum((A - B) ** 2)) * s_ov(a, A, b, B)


def s_nuc(a, A, b, B, Z, C):
    p = a + b
    P = (a * A + b * B) / p
    return (-Z * (2.0 * np.pi / p) * np.exp(-a * b / p * np.sum((A - B) ** 2))
            * F0(p * np.sum((P - C) ** 2)))


def s_eri(a, A, b, B, c, C, d, D):
    p, q = a + b, c + d
    P = (a * A + b * B) / p
    Q = (c * C + d * D) / q
    rho = p * q / (p + q)
    return (2.0 * np.pi ** 2.5 / (p * q * np.sqrt(p + q))
            * np.exp(-a * b / p * np.sum((A - B) ** 2))
            * np.exp(-c * d / q * np.sum((C - D) ** 2))
            * F0(rho * np.sum((P - Q) ** 2)))


def fd1(f, x, i, h=1e-4):
    xp = x.copy(); xp[i] += h
    xm = x.copy(); xm[i] -= h
    return (f(xp) - f(xm)) / (2 * h)


def fd2(f, x, i, j, h=2e-3):
    if i == j:
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        return (f(xp) - 2 * f(x) + f(xm)) / h ** 2
    xpp = x.copy(); xpp[i] += h; xpp[j] += h
    xpm = x.copy(); xpm[i] += h; xpm[j] -= h
    xmp = x.copy(); xmp[i] -= h; xmp[j] += h
    xmm = x.copy(); xmm[i] -= h; xmm[j] -= h
    return (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h ** 2)


class FakeMol:
    def __init__(self, shells, coords, charges):
        self.shells = shells
        self.coords = np.asarray(coords, float)
        self.charges = np.asarray(charges, float)
        self.natom = len(coords)
        self.ncart = sum(s.ncart for s in shells)
        self.nao = sum(s.nsph for s in shells)
        self.enuc = 0.0


def build_test_molecule():
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([1.1, 0.3, -0.4])
    C = np.array([-0.5, 1.3, 0.2])
    D = np.array([0.4, -0.9, 1.2])
    specs = [(A, 0, 1.2), (A, 1, 0.9), (A, 2, 0.6),
             (B, 0, 0.8), (B, 1, 1.1),
             (C, 0, 1.5), (C, 1, 0.5),
             (D, 0, 0.7)]
    shells, cs, ss = [], 0, 0
    for (ctr, l, alpha) in specs:
        sh = Shell(iatom=0, l=l, center=ctr.copy(), exps=np.array([alpha]),
                   coefs=np.array([1.0]), cstart=cs, sstart=ss,
                   transf=cart2sph_matrix(l))
        shells.append(sh); cs += sh.ncart; ss += sh.nsph
    # only the nucleus at C feels a (unit) point charge -> the V reference below
    return FakeMol(shells, [A, B, C, D], [0.0, 0.0, 1.0, 0.0]), (A, B, C, D)


def test_fd():
    mol, (A, B, C, D) = build_test_molecule()
    eng = IntegralEngine(mol)
    S_A, P_A, D_A, S_B, P_B, S_C, P_C, S_D = range(8)
    aA, aPA, aD = 1.2, 0.9, 0.6
    aB, aPB, aC, aPC, aD_ = 0.8, 1.1, 1.5, 0.5, 0.7

    def blk(*idx):
        return eng._eri_block(*idx[:4], sph=False)[idx[4:]]

    def rep(name, got, ref):
        rel = abs(got - ref) / max(abs(ref), 1e-30)
        print(f"  {name:10s} got={got: .10f}  ref={ref: .10f}  rel={rel:.2e}")
        return rel

    errs = []
    print("== 2e: closed form and finite-difference checks ==")
    errs.append(rep("ss|ss", blk(S_A, S_B, S_C, S_D, 0, 0, 0, 0),
                    s_eri(aA, A, aB, B, aC, C, aD_, D)))
    errs.append(rep("ps|ss", blk(P_A, S_B, S_C, S_D, 0, 0, 0, 0),
                    fd1(lambda x: s_eri(aPA, x, aB, B, aC, C, aD_, D), A, 0) / (2 * aPA)))
    errs.append(rep("sp|ss", blk(S_A, P_B, S_C, S_D, 0, 0, 0, 0),
                    fd1(lambda x: s_eri(aA, A, aPB, x, aC, C, aD_, D), B, 0) / (2 * aPB)))
    errs.append(rep("ss|ps", blk(S_A, S_B, P_C, S_D, 0, 0, 0, 0),
                    fd1(lambda x: s_eri(aA, A, aB, B, aPC, x, aD_, D), C, 0) / (2 * aPC)))
    errs.append(rep("dxy|ss", blk(D_A, S_B, S_C, S_D, 1, 0, 0, 0),
                    fd2(lambda x: s_eri(aD, x, aB, B, aC, C, aD_, D), A, 0, 1) / (4 * aD ** 2)))
    errs.append(rep("dzz|ss", blk(D_A, S_B, S_C, S_D, 5, 0, 0, 0),
                    (fd2(lambda x: s_eri(aD, x, aB, B, aC, C, aD_, D), A, 2, 2)
                     + 2 * aD * s_eri(aD, A, aB, B, aC, C, aD_, D)) / (4 * aD ** 2)))

    def mixed_AC(h=2e-3):
        def g(Ax, Cy):
            AA = A.copy(); AA[0] = Ax
            CC = C.copy(); CC[1] = Cy
            return s_eri(aPA, AA, aB, B, aPC, CC, aD_, D)
        return (g(A[0] + h, C[1] + h) - g(A[0] + h, C[1] - h)
                - g(A[0] - h, C[1] + h) + g(A[0] - h, C[1] - h)) / (4 * h ** 2)
    errs.append(rep("ps|ps", blk(P_A, S_B, P_C, S_D, 0, 0, 1, 0),
                    mixed_AC() / (4 * aPA * aPC)))

    print("== 1e: overlap / kinetic / nuclear ==")
    errs.append(rep("S(p)", eng._ov_cart(P_A, S_B)[0, 0],
                    fd1(lambda x: s_ov(aPA, x, aB, B), A, 0) / (2 * aPA)))
    errs.append(rep("T(p)", eng._kin_cart(P_A, S_B)[0, 0],
                    fd1(lambda x: s_kin(aPA, x, aB, B), A, 0) / (2 * aPA)))
    errs.append(rep("V(p)", eng._nuc_cart(P_A, S_B)[0, 0],
                    fd1(lambda x: s_nuc(aPA, x, aB, B, 1.0, C), A, 0) / (2 * aPA)))
    errs.append(rep("S(dxy)", eng._ov_cart(D_A, S_B)[1, 0],
                    fd2(lambda x: s_ov(aD, x, aB, B), A, 0, 1) / (4 * aD ** 2)))
    errs.append(rep("V(dxy)", eng._nuc_cart(D_A, S_B)[1, 0],
                    fd2(lambda x: s_nuc(aD, x, aB, B, 1.0, C), A, 0, 1) / (4 * aD ** 2)))
    return max(errs)


def test_h2():
    """H2/STO-3G closed-shell ROHF energy at R = 1.4 bohr.

    Reference value -1.1167143251 Ha (cross-checked with pyscf's RHF).
    """
    mol = Molecule([("H", (0, 0, 0)), ("H", (0, 0, 1.4))], "STO-3G")
    eng = IntegralEngine(mol)
    S, T, V = eng.one_electron()
    eri = eng.eri()
    scf = ROHF(mol, S, T + V, eri, nocc_c=1, nocc_o=0, max_cycle=200, conv_tol=1e-12)
    e = scf.run()
    print(f"\n== H2 / STO-3G (R = 1.4 bohr) ==\n  E(RHF) = {e:.10f}  "
          f"(reference -1.1167143251, diff {e + 1.1167143251:+.1e})")
    return abs(e + 1.1167143251)


if __name__ == "__main__":
    worst = test_fd()
    h2err = test_h2()
    print(f"\nmax relative error of the integral checks: {worst:.2e}")
    if worst < 1e-4 and h2err < 1e-6:
        print("INTEGRAL/SCF VALIDATION PASSED")
    else:
        print("VALIDATION FAILED")
        sys.exit(1)
