"""Self-contained Hartree-Fock: ROHF (Roothaan effective Fock) and UHF.

Both use a DIIS extrapolation on the commutator error  F D S - S D F  with the
*total* density, and optional damping.  Everything works in the AO basis from
the SCF module's own integrals (no external quantum-chemistry package).

Roothaan ROHF block structure (for a high-spin doublet: one closed shell +
one open shell):

    space     closed   open   virtual
    closed      Fc      Fb      Fc
    open        Fb      Fc      Fa
    virtual     Fc      Fa      Fc

with F^a = h + J[D_tot] - K[D^a],  F^b = h + J[D_tot] - K[D^b],
Fc = (F^a + F^b)/2 and the projectors Pc = D^b S, Po = (D^a-D^b) S,
Pv = 1 - D^a S.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


class DIIS:
    """Direct inversion of the iterative subspace (Pulay)."""

    def __init__(self, nkeep=8, start=2):
        self.nkeep = nkeep
        self.start = start
        self.focks: list[np.ndarray] = []
        self.errors: list[np.ndarray] = []

    def push(self, F, err):
        e = np.asarray(err, dtype=float).ravel()
        nrm = np.linalg.norm(e)
        if nrm > 0:
            e = e / nrm          # scale-free error vectors keep B well conditioned
        self.focks.append(F.copy())
        self.errors.append(e)
        if len(self.focks) > self.nkeep:
            self.focks.pop(0)
            self.errors.pop(0)

    def extrapolate(self):
        m = len(self.focks)
        if m < self.start:
            return None
        B = np.array([[np.dot(a, b) for b in self.errors] for a in self.errors])
        lam = np.zeros((m + 1, m + 1))
        lam[:m, :m] = B
        lam[m, :m] = -1.0
        lam[:m, m] = -1.0
        rhs = np.zeros(m + 1)
        rhs[m] = -1.0
        try:
            c = np.linalg.lstsq(lam, rhs, rcond=None)[0][:m]
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(c)) or np.max(np.abs(c)) > 5.0:
            return None
        F = np.tensordot(c, np.asarray(self.focks), axes=1)
        return 0.5 * (F + F.T)


class _SCFBase:
    def __init__(self, mol, S, h, eri, max_cycle=200, conv_tol=1e-10,
                 diis=True, damping=0.0, verbose=False):
        self.mol = mol
        self.S = S
        self.h = h
        self.eri = eri
        self.max_cycle = max_cycle
        self.conv_tol = conv_tol
        self.use_diis = diis
        self.damping = damping
        self.verbose = verbose

    # -- two-electron pieces -------------------------------------------
    def jk(self, D):
        J = np.einsum("pqrs,rs->pq", self.eri, D)
        K = np.einsum("pqrs,qs->pr", self.eri, D)
        return J, K

    def _diis_update(self, F, D, diis):
        err = F @ (D @ self.S) - (D @ self.S) @ F
        diis.push(F, err)
        Fd = diis.extrapolate()
        return F if Fd is None else Fd

    @staticmethod
    def _damped(F_new, F_old, damping):
        if F_old is None or damping <= 0:
            return F_new
        return (1.0 - damping) * F_new + damping * F_old


class ROHF(_SCFBase):
    """Roothaan restricted open-shell HF for a high-spin doublet."""

    def __init__(self, mol, S, h, eri, nocc_c=1, nocc_o=1, **kw):
        super().__init__(mol, S, h, eri, **kw)
        self.nc = nocc_c
        self.no = nocc_o

    def _densities(self, C, eps):
        order = np.argsort(eps)
        C = C[:, order]
        Cc = C[:, :self.nc]
        Co = C[:, self.nc:self.nc + self.no]
        Db = Cc @ Cc.T                      # beta density (closed shell)
        Do = Co @ Co.T                      # open shell (alpha only)
        return C, Db, Do

    def _fock_ab(self, Da, Db):
        Dtot = Da + Db
        J, _ = self.jk(Dtot)
        _, Ka = self.jk(Da)
        _, Kb = self.jk(Db)
        return self.h + J - Ka, self.h + J - Kb

    def _effective_fock(self, Fa, Fb, Da, Db):
        n = self.S.shape[0]
        Fc = 0.5 * (Fa + Fb)
        Pc = Db @ self.S
        Po = (Da - Db) @ self.S
        Pv = np.eye(n) - Da @ self.S
        F = (0.5 * (Pc.T @ Fc @ Pc) + 0.5 * (Po.T @ Fc @ Po)
             + 0.5 * (Pv.T @ Fc @ Pv)
             + Po.T @ Fb @ Pc + Po.T @ Fa @ Pv + Pv.T @ Fc @ Pc)
        return F + F.T

    def _energy(self, Da, Db, Fa, Fb):
        e = 0.5 * (np.sum(Da * (self.h + Fa)) + np.sum(Db * (self.h + Fb)))
        return e + self.mol.enuc

    def run(self):
        S = self.S
        eps, C = eigh(self.h, S)
        C, Db, Do = self._densities(C, eps)
        Da = Db + Do
        diis = DIIS()
        F_old = None
        E_old = None
        converged = False
        for cycle in range(self.max_cycle):
            Fa, Fb = self._fock_ab(Da, Db)
            E = self._energy(Da, Db, Fa, Fb)
            F = self._effective_fock(Fa, Fb, Da, Db)
            if self.use_diis:
                F = self._diis_update(F, Da + Db, diis)
            F = self._damped(F, F_old, self.damping)
            F_old = F
            eps, C = eigh(F, S)
            C, Db, Do = self._densities(C, eps)
            Da = Db + Do
            if self.verbose:
                print(f"  ROHF {cycle:3d}  E = {E:.12f}  dE = "
                      f"{0.0 if E_old is None else E - E_old:+.2e}")
            if E_old is not None and abs(E - E_old) < self.conv_tol:
                converged = True
                break
            E_old = E
        Fa, Fb = self._fock_ab(Da, Db)
        self.C = C
        self.eps = eps
        self.Da, self.Db, self.Do = Da, Db, Do
        self.Fa, self.Fb = Fa, Fb
        self.energy = self._energy(Da, Db, Fa, Fb)
        self.converged = converged
        self.ncycles = cycle + 1
        self.nocc_c, self.nocc_o = self.nc, self.no
        return self.energy


class UHF(_SCFBase):
    """Unrestricted HF (any number of alpha/beta electrons)."""

    def __init__(self, mol, S, h, eri, nalpha=2, nbeta=1, **kw):
        super().__init__(mol, S, h, eri, **kw)
        self.na = nalpha
        self.nb = nbeta

    def _fock(self, Da, Db):
        Dtot = Da + Db
        J, _ = self.jk(Dtot)
        _, Ka = self.jk(Da)
        _, Kb = self.jk(Db)
        return self.h + J - Ka, self.h + J - Kb

    def _energy(self, Da, Db, Fa, Fb):
        e = 0.5 * (np.sum(Da * (self.h + Fa)) + np.sum(Db * (self.h + Fb)))
        return e + self.mol.enuc

    def run(self):
        S = self.S
        eps, C = eigh(self.h, S)
        Ca = C[:, :self.na].copy()
        Cb = C[:, :self.nb].copy()
        # break the spin symmetry: with a restricted guess the UHF equations
        # have the restricted solution as a fixed point and the lower
        # spin-polarised solution would never be reached
        if self.nb > 0 and C.shape[1] > self.nb:
            v = C[:, self.nb]
            th = 0.2
            Cb[:, -1] = np.cos(th) * Cb[:, -1] + np.sin(th) * v
        Da = Ca @ Ca.T
        Db = Cb @ Cb.T
        diis_a, diis_b = DIIS(), DIIS()
        Fa_old = Fb_old = None
        E_old = None
        converged = False
        for cycle in range(self.max_cycle):
            Fa, Fb = self._fock(Da, Db)
            E = self._energy(Da, Db, Fa, Fb)
            if self.use_diis:
                Fa = self._diis_update(Fa, Da + Db, diis_a)
                Fb = self._diis_update(Fb, Da + Db, diis_b)
            Fa = self._damped(Fa, Fa_old, self.damping)
            Fb = self._damped(Fb, Fb_old, self.damping)
            Fa_old, Fb_old = Fa, Fb
            ea, Ca = eigh(Fa, S)
            eb, Cb = eigh(Fb, S)
            Ca = Ca[:, :self.na]
            Cb = Cb[:, :self.nb]
            Da = Ca @ Ca.T
            Db = Cb @ Cb.T
            if self.verbose:
                print(f"  UHF  {cycle:3d}  E = {E:.12f}  dE = "
                      f"{0.0 if E_old is None else E - E_old:+.2e}")
            if E_old is not None and abs(E - E_old) < self.conv_tol:
                converged = True
                break
            E_old = E
        Fa, Fb = self._fock(Da, Db)
        self.Ca, self.Cb = Ca, Cb
        self.eps_a, self.eps_b = ea, eb
        self.Da, self.Db = Da, Db
        self.energy = self._energy(Da, Db, Fa, Fb)
        self.converged = converged
        self.ncycles = cycle + 1
        return self.energy

    def s2(self):
        """<S^2> of the UHF determinant (S_z = (na-nb)/2)."""
        sz = 0.5 * (self.na - self.nb)
        ov = self.Ca.T @ self.S @ self.Cb          # (na, nb)
        return sz * (sz + 1.0) + self.nb - np.sum(ov ** 2)
