"""h4fci - a self-contained Hartree-Fock + full-CI program for H4+.

Modules
-------
basis     : basis-set parsing (Basis Set Exchange NWChem format) and AO data
integrals : Gaussian one- and two-electron integrals (McMurchie-Davidson)
scf       : ROHF (Roothaan effective Fock) and UHF
fci       : determinant full CI (Slater-Condon + ARPACK / Davidson)

No external quantum-chemistry package is used; only numpy and scipy.
"""
__version__ = "1.0"
