"""Determinant full CI (FCI) for the three-electron doublet of H4+.

Spin symmetry
-------------
H4+ has three electrons; the ground state is a doublet.  Working in the
S_z = +1/2 sector (N_alpha = 2, N_beta = 1) is convenient for three electrons:
the S_z = 1/2 space contains only S = 1/2 states (an S = 3/2 state would need
S_z = +-3/2), so no spin contamination is possible inside this sector.

Representation
--------------
An alpha string with two electrons is a pair (i < j); a beta string with one
electron is a single orbital J.  The CI vector is therefore a matrix

    Psi[I, J]        I : alpha pair index (norb*(norb-1)/2 values)
                     J : beta orbital index (norb values)

which makes every Hamiltonian term either a dense matrix product in the alpha
space (BLAS) or a cheap vectorised update over the beta index.  For
H4+/cc-pVTZ the space is 1540 x 56 = 86 240 determinants.

Slater-Condon rules used (spatial orbitals, (pq|rs) in chemist notation,
spin-orbitals ordered as alpha block then beta block):

    diagonal          sum_p h_pp + sum_{p<q}[(pp|qq) - (pq|qp)]
                      alpha-beta pairs contribute only the Coulomb term
    alpha single      h_ia + sum_{k in I, k != i}[(ia|kk) - (ik|ka)] + (ia|JJ)
    alpha double      (ia|jb) - (ib|ja)
    beta single       h_Jb + sum_{i in I}(Jb|ii)
    alpha-beta double (ia|Jb)

The determinantal sign of an excitation is obtained by applying the ladder
operators one at a time and counting the occupied orbitals below each
operator (``(-1)**(number of occupied orbitals below)``), which is the same
convention as the operator ordering a+_a a+_b a_j a_i.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.linalg import eigh as _geigh

__all__ = ["FCI", "transform_integrals", "determinant_space"]


def transform_integrals(h, eri, C):
    """AO -> MO transformation of one- and two-electron integrals.

    ``eri`` is in chemist notation (pq|rs); the returned tensor is
    ``eri_mo[i,j,k,l] = (ij|kl)`` with MO indices.
    """
    h_mo = C.T @ h @ C
    X = np.tensordot(eri, C, axes=([3], [0]))
    X = np.tensordot(X, C, axes=([2], [0]))
    X = np.tensordot(X, C, axes=([1], [0]))
    X = np.tensordot(X, C, axes=([0], [0]))
    return h_mo, X.transpose(3, 2, 1, 0)


def determinant_space(norb, nalpha=2, nbeta=1):
    """Number of determinants for the (nalpha, nbeta) sector."""
    from math import comb
    return comb(norb, nalpha) * comb(norb, nbeta)


def _popcount_below(mask, p):
    """Number of set bits of ``mask`` strictly below bit ``p``."""
    return bin(mask & ((1 << p) - 1)).count("1")


class FCI:
    """Full CI in the S_z = 1/2 sector for three electrons."""

    def __init__(self, h, eri, nalpha=2, nbeta=1, verbose=False):
        self.h = np.asarray(h, dtype=float)
        self.eri = np.asarray(eri, dtype=float)
        self.norb = self.h.shape[0]
        self.na, self.nb = nalpha, nbeta
        self.verbose = verbose
        if not (self.na == 2 and self.nb == 1):
            raise NotImplementedError("this FCI is specialised to the 3-electron "
                                      "doublet (N_alpha=2, N_beta=1)")
        self._build_strings()
        self._build_operators()

    # ------------------------------------------------------------------
    def _build_strings(self):
        n = self.norb
        pairs = np.array(list(itertools.combinations(range(n), 2)), dtype=np.int64)
        self.a1, self.a2 = pairs[:, 0], pairs[:, 1]
        self.Na = self.a1.size
        self.Nb = n
        self.n_det = self.Na * self.Nb
        self.pair_idx = np.full((n, n), -1, dtype=np.int64)
        idx = np.arange(self.Na)
        self.pair_idx[self.a1, self.a2] = idx
        self.pair_idx[self.a2, self.a1] = idx
        # alpha strings containing orbital p
        self.str_with = [np.nonzero((self.a1 == p) | (self.a2 == p))[0]
                         for p in range(n)]
        self.occ = np.zeros((self.Na, n), dtype=bool)
        self.occ[idx, self.a1] = True
        self.occ[idx, self.a2] = True
        # bit masks (for sign bookkeeping / debugging)
        self.mask = (1 << self.a1) | (1 << self.a2)

    # ------------------------------------------------------------------
    def _build_operators(self):
        n = self.norb
        h, eri = self.h, self.eri
        a1, a2 = self.a1, self.a2
        Na, Nb = self.Na, self.Nb
        ar = np.arange(n)

        # ---- diagonal -------------------------------------------------
        diag_a = (h[a1, a1] + h[a2, a2]
                  + eri[a1, a1, a2, a2] - eri[a1, a2, a2, a1])
        Wc = np.einsum("ppJJ->pJ", eri)               # (pp|JJ)
        cross = self.occ.astype(float) @ Wc
        self.diag = diag_a[:, None] + cross

        # ---- alpha one-body, J independent part -----------------------
        A1 = np.zeros((Na, Na))
        for i in range(n):
            for a in range(n):
                if i == a:
                    continue
                S, T, sg = self._single_alpha_targets(i, a)
                if S.size == 0:
                    continue
                other = np.where(a1[S] == i, a2[S], a1[S])
                me = h[i, a] + eri[i, a, other, other] - eri[i, other, other, a]
                A1[T, S] += sg * me
        self.A1 = A1

        # ---- alpha two-body (double excitations) ---------------------
        A2 = np.zeros((Na, Na))
        all_idx = np.arange(Na)
        for k in range(Na):
            i, j = a1[k], a2[k]
            mask = (all_idx != k) & ~self.occ[all_idx, i] & ~self.occ[all_idx, j]
            T = all_idx[mask]
            if T.size == 0:
                continue
            aa, bb = a1[T], a2[T]
            me = eri[i, aa, j, bb] - eri[i, bb, j, aa]
            A2[T, k] += me
        self.A2 = 0.5 * (A2 + A2.T)

        # ---- precomputed excitation bookkeeping for sigma ------------
        self.a_list = []
        self.tgt_ia = []
        self.sgn_ia = []
        for i in range(n):
            a_vals = np.delete(ar, i)
            S = self.str_with[i]
            others = np.where(a1[S] == i, a2[S], a1[S])
            tgt = np.zeros((a_vals.size, S.size), dtype=np.int64)
            sgn = np.zeros((a_vals.size, S.size))
            for ka, a in enumerate(a_vals):
                valid = ~self.occ[S, a]
                o = others[valid]
                s_ = np.where(((o < i).astype(int) + (o < a).astype(int)) % 2,
                              -1.0, 1.0)
                tgt[ka, valid] = self.pair_idx[np.minimum(o, a), np.maximum(o, a)]
                sgn[ka, valid] = s_
            self.a_list.append(a_vals)
            self.tgt_ia.append(tgt)
            self.sgn_ia.append(sgn)

        # ---- (pp|JJ) folded with the alpha occupation -----------------
        # beta single excitation, alpha spectators: sum_{i in I} (Jb|ii)
        A = eri[:, :, ar, ar]                        # A[p, q, k] = (pq|kk)
        Dall = A.transpose(2, 0, 1)                  # Dall[k, p, q] = (pq|kk)
        self.Msum = Dall[a1] + Dall[a2]              # (Na, n, n) = (Jb|i1i1)+(Jb|i2i2)
        self.h = h

    def _single_alpha_targets(self, i, a):
        """Strings where alpha orbital i is replaced by a (S, T, signs)."""
        S = self.str_with[i]
        S = S[~self.occ[S, a]]
        if S.size == 0:
            return S, S, np.zeros(0)
        other = np.where(self.a1[S] == i, self.a2[S], self.a1[S])
        sg = np.where(((other < i).astype(int) + (other < a).astype(int)) % 2,
                      -1.0, 1.0)
        targ = self.pair_idx[np.minimum(other, a), np.maximum(other, a)]
        return S, targ, sg

    # ------------------------------------------------------------------
    def sigma(self, V):
        """Apply the Hamiltonian to a CI vector of shape (Na, Nb)."""
        n = self.norb
        out = self.diag * V
        out += self.A1 @ V
        out += self.A2 @ V
        out += V @ self.h.T                            # beta one-body
        # beta single excitation with alpha spectators: (Jb|ii), b != J
        out += np.einsum("IJb,IJ->Ib", self.Msum, V, optimize=True)
        out -= np.einsum("Ibb->Ib", self.Msum) * V
        # alpha single (Coulomb part) + alpha-beta double excitation
        # accumulate on ordered (a, other) pairs -- the sorted pair index is
        # reached twice (a,o) and (o,a), so a direct scatter would collide
        tmp = np.zeros((n, n, self.Nb))
        for i in range(n):
            S = self.str_with[i]
            Vs = V[S]
            blocks = np.einsum("sk,akb->asb", Vs, self.eri[i, self.a_list[i]],
                               optimize=True)
            others = np.where(self.a1[S] == i, self.a2[S], self.a1[S])
            vals = self.sgn_ia[i][:, :, None] * blocks
            tmp[self.a_list[i][:, None], others[None, :], :] += vals
        out += tmp[self.a1, self.a2] + tmp[self.a2, self.a1]
        return out

    # ------------------------------------------------------------------
    def davidson(self, nroots=1, tol=1e-9, max_iter=200, max_basis=24,
                 guess=None):
        """Davidson diagonalisation (block, with preconditioning)."""
        nd = self.Na * self.Nb
        diag = self.diag.ravel()
        order = np.argsort(diag)

        basis, Hb = [], []

        def add(v):
            nrm = np.sqrt(np.sum(v * v))
            if nrm < 1e-14:
                return False
            basis.append(v / nrm)
            Hb.append(self.sigma(basis[-1]))
            return True

        if guess is None:
            for r in range(nroots):
                v = np.zeros(nd)
                v[order[r]] = 1.0
                add(v.reshape(self.Na, self.Nb))
        else:
            for r in range(nroots):
                add(np.asarray(guess[r], dtype=float).reshape(self.Na, self.Nb))

        eigs = None
        for it in range(max_iter):
            nb = len(basis)
            Hm = np.zeros((nb, nb))
            Sm = np.zeros((nb, nb))
            for i in range(nb):
                Hm[i, i] = np.sum(basis[i] * Hb[i])
                Sm[i, i] = np.sum(basis[i] * basis[i])
                for j in range(i + 1, nb):
                    Hm[i, j] = Hm[j, i] = np.sum(basis[i] * Hb[j])
                    Sm[i, j] = Sm[j, i] = np.sum(basis[i] * basis[j])
            w, c = _geigh(Hm, Sm)
            nr = min(nroots, nb)
            ritz = [sum(c[k, r] * basis[k] for k in range(nb)) for r in range(nr)]
            # residual from the stored H*basis (no new sigma calls)
            res = []
            for r in range(nr):
                hv = sum(c[k, r] * Hb[k] for k in range(nb))
                res.append(np.linalg.norm(hv - w[r] * ritz[r]))
            eigs = w[:nr]
            if self.verbose and (it % 5 == 0 or max(res) < tol):
                print(f"    FCI iter {it:3d}  "
                      + "  ".join(f"E{r} = {w[r]:.12f}" for r in range(nr))
                      + f"   |res| = {max(res):.1e}")
            if max(res) < tol:
                break
            for r in range(nr):
                hv = sum(c[k, r] * Hb[k] for k in range(nb))
                rr = (hv - w[r] * ritz[r]).ravel()
                denom = w[r] - diag
                denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
                corr = (rr / denom).reshape(self.Na, self.Nb)
                for b in basis:                      # orthogonalise
                    corr -= np.sum(b * corr) * b
                add(corr)
            if len(basis) > max_basis:
                keep = []
                for r in range(nr):
                    v = ritz[r].copy()
                    for b in keep:
                        v -= np.sum(b * v) * b
                    keep.append(v / np.sqrt(np.sum(v * v)))
                basis, Hb = [], []
                for v in keep:
                    add(v)

        # final eigenvalues in the converged subspace
        nb = len(basis)
        Hm = np.zeros((nb, nb))
        Sm = np.zeros((nb, nb))
        for i in range(nb):
            for j in range(nb):
                Hm[i, j] = np.sum(basis[i] * Hb[j])
                Sm[i, j] = np.sum(basis[i] * basis[j])
        w, c = _geigh(Hm, Sm)
        nr = min(nroots, nb)
        self.energies = w[:nr]
        self.vectors = [sum(c[k, r] * basis[k] for k in range(nb)) for r in range(nr)]
        self.n_iter = it + 1
        return self.energies

    # ------------------------------------------------------------------
    def one_rdm(self, vec):
        """Alpha and beta one-particle density matrices in the MO basis."""
        V = np.asarray(vec).reshape(self.Na, self.Nb)
        n = self.norb
        w2 = V * V
        Da = np.zeros((n, n))
        for p in range(n):
            # diagonal: occupation of orbital p in the alpha sector
            Da[p, p] = np.sum(self.occ[:, p] * np.sum(w2, axis=1))
            for q in range(n):
                if p == q:
                    continue
                S, T, sg = self._single_alpha_targets(q, p)
                if S.size:
                    Da[p, q] = np.sum(sg * np.sum(V[T] * V[S], axis=1))
        Db = V.T @ V          # single beta electron: diagonal populations
        return Da, Db

    def natural_occupations(self, vec=None):
        """Natural spin-orbital occupations (descending) of one CI vector."""
        if vec is None:
            vec = self.vectors[0]
        Da, Db = self.one_rdm(vec)
        ea = np.linalg.eigvalsh(Da)[::-1]
        eb = np.linalg.eigvalsh(Db)[::-1]
        return ea, eb
