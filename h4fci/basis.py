"""Basis-set data handling for the H4+ FCI program.

Basis sets are read from the NWChem-format files in ``h4fci/data`` which were
downloaded from the Basis Set Exchange (https://www.basissetexchange.org).

Conventions
-----------
* Cartesian component order is lexicographic (the order used by pyscf):

      l=0 : s
      l=1 : x, y, z
      l=2 : xx, xy, xz, yy, yz, zz

* The one-electron basis used by the SCF and CI is *spherical*: 1 function for
  s, 3 for p, 5 for d.  The 6 cartesian -> 5 spherical transformation for d is

      d(xy)      = sqrt(3) xy
      d(yz)      = sqrt(3) yz
      d(z^2)     = (2 zz - xx - yy)/2
      d(xz)      = sqrt(3) xz
      d(x^2-y^2) = sqrt(3)/2 (xx - yy)

  with the components stored in that order (m = -2, -1, 0, +1, +2), which is
  the ordering pyscf uses for spherical functions.  All five spherical d
  functions have unit norm with respect to the (normalised) cartesian basis.

* Contracted shells are normalised so that every cartesian component has unit
  self overlap, i.e. the usual quantum-chemistry normalisation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# angular powers (ax, ay, az) for the cartesian components, pyscf ordering
CART_POWERS = {
    0: ((0, 0, 0),),
    1: ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    2: ((2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)),
    3: ((3, 0, 0), (2, 1, 0), (2, 0, 1), (1, 2, 0), (1, 1, 1),
        (1, 0, 2), (0, 3, 0), (0, 2, 1), (0, 1, 2), (0, 0, 3)),
}

_L_OF = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4, "H": 5, "I": 6}

ELEMENTS = {
    "H": (1, 1.0, "H"),
    "He": (2, 2.0, "He"),
}


def ncart(l: int) -> int:
    """Number of cartesian components of angular momentum l."""
    return (l + 1) * (l + 2) // 2


def nsph(l: int) -> int:
    """Number of spherical components of angular momentum l."""
    return 2 * l + 1


def cart2sph_matrix(l: int) -> np.ndarray:
    """(ncart, nsph) matrix taking normalised cartesians to normalised spheres."""
    if l == 0:
        return np.ones((1, 1))
    if l == 1:
        return np.eye(3)
    if l == 2:
        s3 = np.sqrt(3.0)
        # order of the 5 rows: xy, yz, z^2, xz, x^2-y^2
        return np.array([
            [0.0,    s3,  0.0, 0.0,   0.0, 0.0],   # xy
            [0.0,    0.0, 0.0, 0.0,    s3, 0.0],   # yz
            [-0.5,   0.0, 0.0, -0.5,   0.0, 1.0],  # z^2
            [0.0,    0.0, s3,  0.0,    0.0, 0.0],  # xz
            [0.5 * s3, 0.0, 0.0, -0.5 * s3, 0.0, 0.0],  # x^2-y^2
        ])
    raise NotImplementedError(f"spherical transformation not implemented for l={l}")


@dataclass
class Shell:
    """One *contracted* shell (a single contraction, already normalised)."""

    iatom: int
    l: int
    center: np.ndarray                    # (3,) in bohr
    exps: np.ndarray                      # (nprim,)
    coefs: np.ndarray                     # (nprim,) contraction coefficients
    cstart: int                           # first cartesian function index
    sstart: int                           # first spherical function index
    transf: np.ndarray = field(default=None)   # (ncart(l), nsph(l))

    @property
    def ncart(self) -> int:
        return ncart(self.l)

    @property
    def nsph(self) -> int:
        return nsph(self.l)


def parse_nwchem(path: str) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Parse a BSE NWChem-format file for one element.

    Returns a list of ``(l, exponents, coefficients)`` where ``coefficients``
    has shape ``(nprim, ncontr)``.
    """
    blocks: list[dict] = []
    cur = None
    with open(path) as fh:
        for raw in fh:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            up = s.upper()
            if up.startswith("BASIS") or up.startswith("END"):
                continue
            toks = s.split()
            # shell header: "<element> <L>" with a single angular-momentum letter
            if (len(toks) == 2 and toks[0].isalpha()
                    and len(toks[1]) == 1 and toks[1].upper() in _L_OF):
                cur = {"l": _L_OF[toks[1].upper()], "rows": []}
                blocks.append(cur)
                continue
            if cur is None:
                continue
            vals = [float(t.replace("D", "E").replace("d", "e")) for t in toks]
            cur["rows"].append(vals)

    out = []
    for blk in blocks:
        rows = np.array(blk["rows"], dtype=float)
        nprim = rows.shape[0]
        ncontr = rows.shape[1] - 1
        exps = rows[:, 0].copy()
        coefs = rows[:, 1:].copy()
        assert ncontr >= 1
        # drop primitives that are not used by any contraction
        keep = np.any(coefs != 0.0, axis=1)
        out.append((blk["l"], exps[keep], coefs[keep]))
    return out


def _prim_norm(l: int, exps: np.ndarray) -> np.ndarray:
    """Normalisation constant of a primitive cartesian gaussian (any component)."""
    # N = (2a/pi)^{3/4} (4a)^{l/2} / sqrt((2l-1)!!)
    from math import factorial
    dfact = 1.0
    for k in range(2 * l - 1, 0, -2):
        dfact *= k
    return (2.0 * exps / np.pi) ** 0.75 * (4.0 * exps) ** (0.5 * l) / np.sqrt(dfact)


def _contract_norm(l: int, exps: np.ndarray, coefs: np.ndarray) -> float:
    """Factor normalising a contracted shell to unit self-overlap."""
    a = exps[:, None]
    b = exps[None, :]
    zeta = a + b
    from math import factorial
    dfact = 1.0
    for k in range(2 * l - 1, 0, -2):
        dfact *= k
    # <g_a|g_b> for unit-normalised primitives of the same shell, same centre
    ov = dfact / (2.0 * zeta) ** l * (np.pi / zeta) ** 1.5
    val = coefs @ ov @ coefs
    return 1.0 / np.sqrt(val)


class Molecule:
    """Molecule = atoms + contracted shells + basis-function bookkeeping."""

    def __init__(self, atoms, basis: str = "cc-pVTZ"):
        # atoms: iterable of (symbol, (x, y, z)) with coordinates in bohr
        self.atoms = [(str(sym), np.asarray(xyz, dtype=float)) for sym, xyz in atoms]
        self.basis_name = basis
        self.symbols = [a[0] for a in self.atoms]
        self.coords = np.array([a[1] for a in self.atoms])
        self.charges = np.array([ELEMENTS[s][1] for s in self.symbols])
        self.natom = len(self.atoms)

        shells: list[Shell] = []
        cstart = 0
        sstart = 0
        cache: dict[str, list] = {}
        for iatom, sym in enumerate(self.symbols):
            key = sym
            if key not in cache:
                path = os.path.join(DATA_DIR, f"{sym}.{basis}.nwchem")
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"no basis data for {sym} in {basis}: {path}")
                cache[key] = parse_nwchem(path)
            for (l, exps, coef2d) in cache[key]:
                for ic in range(coef2d.shape[1]):
                    cf = coef2d[:, ic].copy()
                    nz = cf != 0.0
                    e = exps[nz]
                    # basis-set-exchange coefficients multiply *normalised*
                    # primitive gaussians; convert to the raw-primitive
                    # convention used by the integral engine
                    c = cf[nz] * _prim_norm(l, e)
                    c *= _contract_norm(l, e, c)
                    shells.append(Shell(
                        iatom=iatom, l=l, center=self.coords[iatom].copy(),
                        exps=e, coefs=c, cstart=cstart, sstart=sstart,
                        transf=cart2sph_matrix(l)))
                    cstart += ncart(l)
                    sstart += nsph(l)
        self.shells = shells
        self.ncart = cstart
        self.nao = sstart

        # nuclear repulsion energy
        enuc = 0.0
        for i in range(self.natom):
            for j in range(i + 1, self.natom):
                r = np.linalg.norm(self.coords[i] - self.coords[j])
                enuc += self.charges[i] * self.charges[j] / r
        self.enuc = enuc

    # -- convenience -----------------------------------------------------
    def ao_labels(self) -> list[str]:
        labels = []
        for sh in self.shells:
            for m in range(sh.nsph):
                if sh.l == 0:
                    name = "s"
                elif sh.l == 1:
                    name = "px py pz".split()[m]
                elif sh.l == 2:
                    name = "dxy dyz dz2 dxz dx2-y2".split()[m]
                else:
                    name = f"l{sh.l}m{m}"
                labels.append(f"{sh.iatom} {self.symbols[sh.iatom]} {name}")
        return labels

    def __repr__(self):
        return (f"<Molecule {''.join(self.symbols)} basis={self.basis_name} "
                f"nao={self.nao} nshell={len(self.shells)} Enuc={self.enuc:.6f}>")
