"""
generate_BGWQE_inputs.py

Generates Quantum ESPRESSO + BerkeleyGW input files for an ensemble of
thermally displaced (and optionally strained) structures.

Configuration is read from parameters.py.
Atomic positions and lattice vectors are read from ATOMS.dat.

Usage:
    python generate_BGWQE_inputs.py [--atoms ATOMS.dat] [--params parameters.py]
                                    [--nreplicas N] [--delta D]
                                    [--eps_xx E] [--eps_yy E] [--eps_xy E]
                                    [--seed S] [--outdir replicas]
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
from ase import Atoms
from ase.io import write


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_parameters(params_file):
    spec = importlib.util.spec_from_file_location("parameters", params_file)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_atoms_dat(filename):
    """
    Read lattice vectors and atomic positions from ATOMS.dat.

    Format:
        line 1-3: lattice vectors (Angstrom), three floats per line
        line 4+:  symbol  x  y  z  (Angstrom, Cartesian); lines starting
                  with '#' are ignored everywhere.
    """
    cell      = []
    symbols   = []
    positions = []

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) == 3 and len(cell) < 3:
                cell.append([float(x) for x in parts])
            elif len(parts) == 4:
                symbols.append(parts[0])
                positions.append([float(x) for x in parts[1:]])

    if len(cell) != 3:
        raise ValueError(f"Expected 3 lattice vectors in {filename}, found {len(cell)}")
    if not symbols:
        raise ValueError(f"No atomic positions found in {filename}")

    return np.array(cell), symbols, np.array(positions)


# ── Structure transformations ──────────────────────────────────────────────────

def apply_strain(atoms, eps_xx, eps_yy, eps_xy):
    """Apply a 2D strain tensor (eps_xx, eps_yy, eps_xy) to cell and positions."""
    F = np.array([
        [1.0 + eps_xx, eps_xy / 2.0, 0.0],
        [eps_xy / 2.0, 1.0 + eps_yy, 0.0],
        [0.0,          0.0,           1.0],
    ])
    strained = atoms.copy()
    strained.set_cell(atoms.get_cell() @ F.T)
    strained.set_positions(atoms.get_positions() @ F.T)
    return strained


def apply_displacements(atoms, delta, rng):
    """Apply Gaussian random displacements with std=delta (Angstrom)."""
    displaced = atoms.copy()
    if delta > 0.0:
        disp = rng.normal(loc=0.0, scale=delta, size=(len(atoms), 3))
        displaced.positions += disp
    return displaced


# ── QE input writers ──────────────────────────────────────────────────────────

def generate_kpoints(Nkx, Nky, Nkz, qx, qy, qz):
    lines = [f"K_POINTS crystal", f"{Nkx * Nky * Nkz}"]
    for ik in range(Nkx):
        for jk in range(Nky):
            for kk in range(Nkz):
                lines.append(f"{ik/Nkx + qx:.8f}  {jk/Nky + qy:.8f}  {kk/Nkz + qz:.8f}  1.0")
    return "\n".join(lines) + "\n"


def build_qe_dict(calc_type, p):
    return {
        "control": {
            "calculation":  calc_type,
            "prefix":       p.PREFIX,
            "pseudo_dir":   p.pseudodir,
            "verbosity":    "high",
            "tstress":      True,
            "tprnfor":      True,
            "outdir":       "./",
        },
        "system": {
            "ecutwfc":      p.E_cutoff,
            "nbnd":         p.nbnds,
            "occupations":  "smearing",
            "smearing":     "gaussian",
            "degauss":      0.01,
        },
        "electrons": {
            "conv_thr":           p.conv_thr,
            "diagonalization":    "david",
            "diago_david_ndim":   4,
            "diago_full_acc":     True,
        },
    }


def write_qe_input(atoms, path, calc_type, p, kpoints):
    """Write a QE input file, replacing the ASE-generated K_POINTS block."""
    write(path, atoms, format="espresso-in",
          pseudopotentials=p.pseudopotentials,
          input_data=build_qe_dict(calc_type, p))

    with open(path) as f:
        lines = [l for l in f if "K_POINTS" not in l]
    with open(path, "w") as f:
        f.writelines(lines)
        f.write(kpoints)

    print(f"  Written: {path}")


# ── BGW input writers ─────────────────────────────────────────────────────────

def write_pw2bgw(path, Nk, dk1=0.0):
    with open(path, "w") as f:
        f.write(f"""&input_pw2bgw
  prefix        = 'mos2'
  real_or_complex = 2
  wfng_flag     = .true.
  wfng_file     = 'wfn.complex'
  wfng_kgrid    = .true.
  wfng_nk1      = {Nk}
  wfng_nk2      = {Nk}
  wfng_nk3      = 1
  wfng_dk1      = {dk1}
  wfng_dk2      = 0
  wfng_dk3      = 0
/
""")
    print(f"  Written: {path}")


def write_pw2bgw_wfn(path, Nk, dk1=0.0):
    """pw2bgw for 2-wfn_co: also exports RHO, VXC, VSC, VKB."""
    with open(path, "w") as f:
        f.write(f"""&input_pw2bgw
  prefix        = 'mos2'
  real_or_complex = 2
  wfng_flag     = .true.
  wfng_file     = 'wfn.complex'
  wfng_kgrid    = .true.
  wfng_nk1      = {Nk}
  wfng_nk2      = {Nk}
  wfng_nk3      = 1
  wfng_dk1      = {dk1}
  wfng_dk2      = 0
  wfng_dk3      = 0
  rhog_flag     = .true.
  rhog_file     = 'RHO'
  vxcg_flag     = .true.
  vxcg_file     = 'VXC'
  vxc_flag      = .false.
  vscg_flag     = .true.
  vscg_file     = 'VSC'
  vkbg_flag     = .true.
  vkbg_file     = 'VKB'
/
""")
    print(f"  Written: {path}")


def write_projwfc(path):
    with open(path, "w") as f:
        f.write("""&PROJWFC
   outdir          = './'
   prefix          = 'mos2'
   lsym            = .true.
   lwrite_overlaps = .true.
/
""")
    print(f"  Written: {path}")


def write_parabands(path, p):
    with open(path, "w") as f:
        f.write(f"""input_wfn_file  wfn.complex
output_wfn_file WFN.h5
vsc_file        VSC
vkb_file        VKB
verbosity       2

use_pseudobands
protected_cond_bands {p.protected_cond_bands}
accumulation_window  {p.accumulation_window}
""")
    print(f"  Written: {path}")


def write_epsilon(path, Nk, p):
    kpts = "begin qpoints\n"
    for ik in range(Nk):
        for jk in range(Nk):
            if ik == 0 and jk == 0:
                kpts += "0.001       0.000       0.000   1.0  1\n"
            else:
                kpts += f"{ik/Nk:.8f} {jk/Nk:.8f} 0.000   1.0  0\n"
    kpts += "end\n"

    with open(path, "w") as f:
        f.write(f"""use_wfn_hdf5
verbosity 3
epsilon_cutoff {p.epsilon_cutoff}
cell_slab_truncation
degeneracy_check_override

{kpts}""")
    print(f"  Written: {path}")


def write_sigma(path, Nk, p):
    kpts = "begin kpoints\n"
    for ik in range(Nk):
        for jk in range(Nk):
            kpts += f"{ik/Nk:.8f} {jk/Nk:.8f} 0.000   1.0\n"
    kpts += "end\n"

    with open(path, "w") as f:
        f.write(f"""verbosity 3
use_wfn_hdf5
degeneracy_check_override
cell_slab_truncation
band_index_min {p.band_index_min}
band_index_max {p.band_index_max}
dont_use_vxcdat
screening_semiconductor
no_symmetries_q_grid

{kpts}""")
    print(f"  Written: {path}")


# ── Main generation loop ───────────────────────────────────────────────────────

def generate_replicas(atoms_base, p, Nreplicas, delta, eps_xx, eps_yy, eps_xy,
                      seed, base_dir):
    rng = np.random.default_rng(seed)

    kpoints_co  = generate_kpoints(p.Nk_co, p.Nk_co, 1, 0.0,   0.0, 0.0)
    kpoints_qco = generate_kpoints(p.Nk_co, p.Nk_co, 1, 1e-3,  0.0, 0.0)

    # Apply strain once (same for all replicas)
    atoms_strained = apply_strain(atoms_base, eps_xx, eps_yy, eps_xy)

    os.makedirs(base_dir, exist_ok=True)

    for i in range(Nreplicas):
        rep_dir = os.path.join(base_dir, f"replica_{i:03d}")
        print(f"\nGenerating {rep_dir}")

        for sub in ["1-scf", "2-wfn_co", "3-wfnq_co", "4-parabands", "5-epsilon", "6-sigma"]:
            os.makedirs(os.path.join(rep_dir, sub), exist_ok=True)

        atoms = apply_displacements(atoms_strained, delta, rng)

        # QE inputs
        write_qe_input(atoms, f"{rep_dir}/1-scf/scf.in",         "scf",   p, kpoints_co)
        write_pw2bgw(          f"{rep_dir}/1-scf/pw2bgw.in",      p.Nk_co)

        write_qe_input(atoms, f"{rep_dir}/2-wfn_co/bands.in",    "bands", p, kpoints_co)
        write_pw2bgw_wfn(      f"{rep_dir}/2-wfn_co/pw2bgw.in",  p.Nk_co)
        write_projwfc(         f"{rep_dir}/2-wfn_co/projwfc.in")

        write_qe_input(atoms, f"{rep_dir}/3-wfnq_co/bands.in",   "bands", p, kpoints_qco)
        write_pw2bgw(          f"{rep_dir}/3-wfnq_co/pw2bgw.in", p.Nk_co, dk1=1e-3)

        # BGW inputs
        write_parabands(f"{rep_dir}/4-parabands/parabands.inp", p)
        write_epsilon(  f"{rep_dir}/5-epsilon/epsilon.inp",     p.Nk_co, p)
        write_sigma(    f"{rep_dir}/6-sigma/sigma.inp",         p.Nk_co, p)

    print(f"\nDone. {Nreplicas} replicas written to '{base_dir}/'")


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Generate QE + BGW input files for displaced/strained structures")
    parser.add_argument("--atoms",      default="ATOMS.dat",      help="Atomic positions file")
    parser.add_argument("--params",     default="parameters.py",  help="Parameters file")
    parser.add_argument("--nreplicas",  type=int,   default=None, help="Number of replicas (overrides parameters.py)")
    parser.add_argument("--delta",      type=float, default=None, help="Displacement magnitude in Angstrom (overrides parameters.py)")
    parser.add_argument("--eps_xx",     type=float, default=None, help="Strain eps_xx (overrides parameters.py)")
    parser.add_argument("--eps_yy",     type=float, default=None, help="Strain eps_yy (overrides parameters.py)")
    parser.add_argument("--eps_xy",     type=float, default=None, help="Strain eps_xy (overrides parameters.py)")
    parser.add_argument("--seed",       type=int,   default=None, help="Random seed (overrides parameters.py)")
    parser.add_argument("--outdir",     default="replicas",       help="Output base directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    p = load_parameters(args.params)

    cell, symbols, positions = read_atoms_dat(args.atoms)
    atoms_base = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

    Nreplicas = args.nreplicas if args.nreplicas is not None else p.Nreplicas
    delta     = args.delta     if args.delta     is not None else p.delta
    eps_xx    = args.eps_xx    if args.eps_xx    is not None else p.eps_xx
    eps_yy    = args.eps_yy    if args.eps_yy    is not None else p.eps_yy
    eps_xy    = args.eps_xy    if args.eps_xy    is not None else p.eps_xy
    seed      = args.seed      if args.seed      is not None else p.seed

    print(f"Atoms file:   {args.atoms}  ({len(symbols)} atoms)")
    print(f"Parameters:   {args.params}")
    print(f"Nreplicas:    {Nreplicas}")
    print(f"delta:        {delta} Ang")
    print(f"Strain:       eps_xx={eps_xx}, eps_yy={eps_yy}, eps_xy={eps_xy}")
    print(f"Seed:         {seed}")
    print(f"Output dir:   {args.outdir}/")

    generate_replicas(atoms_base, p, Nreplicas, delta, eps_xx, eps_yy, eps_xy,
                      seed, args.outdir)
