# pre_proc — Preprocessing Scripts

These two scripts convert raw Quantum ESPRESSO + BerkeleyGW output into the HDF5 dataset format consumed by the GNN. They must be run in order (Step 1 then Step 2) for each new structure.

---

## Step 1 — `map_orbitals_atoms.py`

Parses `projwfc.out` and builds a mapping from the full set of atomic wavefunctions to a reduced irreducible set. Orbitals that are equivalent by symmetry (same atom type, same `s`, `l`, `m` quantum numbers) are collapsed into a single reduced index. This mapping is required by Step 2.

### Usage

```bash
python map_orbitals_atoms.py -projwfc_output projwfc.out
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-projwfc_output` / `--projwfc_out_file` | `projwfc.out` | Output file from `projwfc.x` (Quantum ESPRESSO) |

### Output

- `orbital_mapping.txt` — three-column text file: `atom_index  original_orbital_index  reduced_orbital_index`

---

## Step 2 — `get_proj_for_graphs_and_eqp.py`

Reads the wavefunction projections (`atomic_proj.xml`), the GW quasiparticle energies (`eqp.dat`), the orbital mapping from Step 1, and the QE input file for geometry. Merges everything into a single HDF5 file ready for the GNN.

### Usage

```bash
python get_proj_for_graphs_and_eqp.py \
    -eqp eqp.dat \
    -Nval <valence_band_index> \
    -proj_file atomic_proj.xml \
    -orbital_mapping_file orbital_mapping.txt \
    -qe_input_file qe.in \
    -output data.h5
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-eqp` / `--eqp_file` | `eqp.dat` | QP energy file from BerkeleyGW sigma step |
| `-Nval` / `--Nval` | `0` | Band index of the top valence band (used to set the energy zero). Must match the band numbering in `eqp.dat` |
| `-proj_file` / `--proj_file` | `atomic_proj.xml` | Wavefunction projection file from `projwfc.x` |
| `-orbital_mapping_file` / `--orbital_mapping_file` | `orbital_mapping.txt` | Output of `map_orbitals_atoms.py` |
| `-qe_input_file` / `--qe_input_file` | `qe.in` | Quantum ESPRESSO input file (used to read atomic positions and lattice vectors) |
| `-output` / `--output_file` | `data.h5` | Output HDF5 filename |
| `-plot_data` / `--plot_data` | `False` | If True, saves a scatter plot of `Edft` vs `Eqp - Edft` |
| `-dont_use_eqp` / `--dont_use_eqp` | `False` | If True, skips `eqp.dat` and sets all QP corrections to zero. Useful for training on DFT-only data or testing the pipeline without GW output |

### Output HDF5 structure

| Dataset | Shape | Description |
|---------|-------|-------------|
| `atom_orb_projections` | `(Nk, Nb, Natoms, Norbs)` | Orbital projections $\|{\langle\phi_{Ro}\|\psi_{nk}\rangle}\|^2$ |
| `Edft` | `(Nb, Nk)` | DFT eigenvalues in eV, zeroed at top of valence band |
| `qp_corrections` | `(Nb, Nk)` | GW corrections $\Delta E_{nk} = E^{\rm QP} - E^{\rm DFT}$ in eV |
| `atomic_positions` | `(Natoms, 3)` | Cartesian positions in Å |
| `lattice_vectors` | `(3, 3)` | Lattice matrix in Å |
| `atomic_species` | `(Natoms,)` | Element symbols (e.g. `['Mo', 'S', 'S']`) |

### Data list file

After generating HDF5 files for all structures, create a plain text file listing them one per line:

```
/path/to/data_structure_1.h5
/path/to/data_structure_2.h5
...
```

This file is passed to the training and optimization scripts via `--file_list_data`.
