

from ase import Atoms
from ase.io import read, write
import numpy as np
import os
import subprocess

PREFIX = 'mos2'
E_cutoff = 35
Nk_co = 6
nbnds = 30 # number of electrons = 26
conv_thr = 1e-8
pseudodir = '/home/rrodriguesdelgrand/scratch/pseudos/nc-sr-04_pw_standard/'
pseudopotentials = {
    "Mo": "Mo.upf",
    "S": "S.upf"
}




def generate_kpoints(Nkx, Nky, Nkz, qx, qy, qz):

    Nk_tot = Nkx * Nky * Nkz

    text = "K_POINTS crystal\n"
    text += f"{Nk_tot}\n"

    for ik in range(Nkx):
        for jk in range(Nky):
            for kk in range(Nkz):
                text += f"{(ik/Nkx+qx):.8f}  {(jk/Nky+qy):.8f}  {(kk/Nkz+qz):.8f}  1.0\n"
    return text

def write_kpoints_qe(input_file, kpoints):
    with open(input_file, 'a') as f:
        f.write(kpoints)

def dictionary_calc(calc_type, nbnds):
    calc_dict = {
            "control": {
                "calculation": calc_type,
                "prefix": PREFIX,
                "pseudo_dir": pseudodir,
                "verbosity": "high",
                "tstress": True,
                "tprnfor": True,
                "outdir": "./"
            },
            "system": {
                "ecutwfc": E_cutoff,
                "nbnd": nbnds,
                "occupations": "smearing",
                "smearing": "gaussian",
                "degauss": 0.01
            },
            "electrons": {
                "conv_thr": conv_thr,
                'diagonalization': 'david',
                'diago_david_ndim': 4,
                'diago_full_acc': True
            }
    }

    return calc_dict

def write_qe_input_file(atoms, qe_input_file, calc_type, nbnds, kpoints):
    print("Writing QE input file:", qe_input_file)
    calc_dict_scf = dictionary_calc(calc_type, nbnds)

    write(qe_input_file, atoms, format="espresso-in", pseudopotentials=pseudopotentials, input_data=calc_dict_scf)

    # Remove any line containing "K_POINTS" from the file
    with open(qe_input_file, 'r') as f:
        lines = f.readlines()

    with open(qe_input_file, 'w') as f:
        for line in lines:
            if "K_POINTS" not in line:
                f.write(line)

    with open(qe_input_file, 'a') as f:
        f.write(kpoints)

    print("Finished writing QE input:", qe_input_file)


# Define unit cell
cell = [
    [ 3.120330788,  0.000000000,  0.000000000],
    [-1.560165395,  2.702285733,  0.000000000],
    [ 0.000000000,  0.000000000, 25.000000000]
]

# Atomic positions
symbols = ['Mo', 'S', 'S']
positions = [
    [0.0000000000, 0.0000000000, 0.0000000000],
    [0.0000000000, 1.8015238204, 1.5533646052],
    [0.0000000000, 1.8015238204, -1.5533646052]
]

# Create ASE Atoms object
atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

# Make 2x2x1 supercell
supercell = atoms.repeat((2, 2, 1))

# kpoints
Nkx, Nky, Nkz = Nk_co, Nk_co, 1
qx, qy, qz = 0, 0, 0
kpoints_co = generate_kpoints(Nkx, Nky, Nkz, qx, qy, qz)

Nkx, Nky, Nkz = Nk_co, Nk_co, 1
qx, qy, qz = 1e-3, 0, 0
kpoints_qco = generate_kpoints(Nkx, Nky, Nkz, qx, qy, qz)

# Write SCF input file
def write_qe_inputs_2x2MoS2(atoms, dir):
    write_qe_input_file(atoms, dir+'/1-scf/scf.in', "scf", nbnds, kpoints_co)
    write_qe_input_file(atoms, dir+'/2-wfn_co/bands.in', "bands", nbnds, kpoints_co)
    write_qe_input_file(atoms, dir+'/3-wfnq_co/bands.in', "bands", nbnds, kpoints_qco)

def write_parabands(parabands_input):
    arq = open(parabands_input, 'w')
    arq.write('''
input_wfn_file wfn.complex
output_wfn_file WFN.h5
vsc_file VSC
vkb_file VKB
verbosity 2

use_pseudobands
protected_cond_bands 40
accumulation_window 0.02''')
    arq.close()
    print('Finished writing parabands input:', parabands_input)

def write_epsilon(epsilon_input, Nk):
    arq = open(epsilon_input, 'w')
    arq.write('''
use_wfn_hdf5
verbosity 3
epsilon_cutoff 10.0
cell_slab_truncation
degeneracy_check_override
              ''')
    kpoints_text = '\nbegin qpoints\n'
    for ik in range(Nk):
        for jk in range(Nk):
            if ik == 0 and jk == 0:
                kpoints_text += f'0.001 0.000 0.000  1.0  1\n'
            else:
                kpoints_text += f'{(ik/Nk):.8f} {(jk/Nk):.8f} 0.000000  1.0 0\n'
    kpoints_text += 'end\n'
    arq.write(kpoints_text)
    arq.close()
    print('Finished writing epsilon input:', epsilon_input)


def write_sigma(sigma_input, Nk):
    arq = open(sigma_input, 'w')
    arq.write('''
verbosity 3
use_wfn_hdf5
degeneracy_check_override
cell_slab_truncation
band_index_min 9
band_index_max 20
dont_use_vxcdat
screening_semiconductor
no_symmetries_q_grid
              ''')
    kpoints_text = '\nbegin kpoints\n'
    for ik in range(Nk):
        for jk in range(Nk):
            kpoints_text += f'{(ik/Nk):.8f} {(jk/Nk):.8f} 0.000000  1.0\n'
    kpoints_text += 'end\n'
    arq.write(kpoints_text)
    arq.close()
    print('Finished writing epsilon input:', sigma_input)

def generate_displaced_structures(atoms, Nreplicas=10, sigma=0.1, seed=42, base_dir="replicas"):
    np.random.seed(seed)
    os.makedirs(base_dir, exist_ok=True)

    for i in range(Nreplicas):
        displaced_atoms = atoms.copy()
        displacements = np.random.normal(loc=0.0, scale=sigma, size=(len(displaced_atoms), 3))
        displaced_atoms.positions += displacements

        # Directory for replica
        rep_dir = os.path.join(base_dir, f"replica_{i:03d}")
        os.makedirs(rep_dir, exist_ok=True)

        # Create subdirectories for scf, wfn_co, wfnq_co
        for sub in ['1-scf', '2-wfn_co', '3-wfnq_co', '4-parabands', '5-epsilon', '6-sigma']:
            os.makedirs(os.path.join(rep_dir, sub), exist_ok=True)

        # Write QE input files
        write_qe_inputs_2x2MoS2(displaced_atoms, rep_dir)

        # write BGW files
        write_parabands(os.path.join(rep_dir, '4-parabands/parabands.inp'))
        write_epsilon(os.path.join(rep_dir, '5-epsilon/epsilon.inp'), Nk_co)
        write_sigma(os.path.join(rep_dir, '6-sigma/sigma.inp'), Nk_co)


generate_displaced_structures(atoms, Nreplicas=50, sigma=0.1, seed=42, base_dir="replicas")
