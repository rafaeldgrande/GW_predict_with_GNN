# data_preparation

Scripts to generate Quantum ESPRESSO + BerkeleyGW input files for an ensemble
of thermally displaced and/or strained structures, for use as GNN training data.

---

## Files

| File | Description |
|---|---|
| `generate_BGWQE_inputs.py` | Main script — generates all QE and BGW input files |
| `parameters.py` | All QE and BGW parameters (edit this to configure a run) |
| `ATOMS.dat` | Reference atomic structure (lattice vectors + positions) |

---

## How it works

`generate_BGWQE_inputs.py` reads a reference structure from `ATOMS.dat` and
generates `Nreplicas` directories, each containing a complete set of QE + BGW
input files for a displaced (and optionally strained) geometry.

For each replica, in order:

1. **Strain** — a uniform strain tensor `(eps_xx, eps_yy, eps_xy)` is applied
   to the lattice vectors and atomic positions. The same strain is used for all
   replicas (useful for studying strain dependence). Default is 0 (no strain).

2. **Random displacements** — Gaussian random displacements with standard
   deviation `delta` (in Angstrom) are applied independently to each atom and
   each Cartesian direction. Default is 0.1 Å.

3. **Input files** are written into the directory structure:

```
replicas/
└── replica_000/
    ├── 1-scf/
    │   ├── scf.in        QE SCF input
    │   └── pw2bgw.in     Convert SCF WFN to BGW format
    ├── 2-wfn_co/
    │   ├── bands.in      QE bands on GW k-grid (unshifted)
    │   ├── pw2bgw.in     Convert WFN + export RHO/VXC/VSC/VKB
    │   └── projwfc.in    Wavefunction projections (for GNN preprocessing)
    ├── 3-wfnq_co/
    │   ├── bands.in      QE bands on shifted k-grid (dk1 = +0.001)
    │   └── pw2bgw.in     Convert shifted WFN
    ├── 4-parabands/
    │   └── parabands.inp Generate pseudo bands
    ├── 5-epsilon/
    │   └── epsilon.inp   BerkeleyGW dielectric matrix
    └── 6-sigma/
        └── sigma.inp     BerkeleyGW self-energy (QP corrections → eqp.dat)
```

---

## ATOMS.dat format

```
# Lines starting with # are ignored
a1x  a1y  a1z        <- lattice vector 1 (Angstrom)
a2x  a2y  a2z        <- lattice vector 2
a3x  a3y  a3z        <- lattice vector 3
Mo   x  y  z         <- atomic positions (symbol + Cartesian coordinates, Angstrom)
S    x  y  z
...
```

---

## parameters.py

Edit `parameters.py` to configure the calculation. Key parameters:

**Quantum ESPRESSO:**

| Parameter | Description |
|---|---|
| `PREFIX` | QE prefix (output file names) |
| `E_cutoff` | Plane-wave cutoff (Ry) |
| `Nk_co` | k-grid size: `Nk_co × Nk_co × 1` |
| `nbnds` | Number of bands |
| `conv_thr` | SCF convergence threshold |
| `pseudodir` | Path to pseudopotential directory |
| `pseudopotentials` | Dict mapping element → pseudopotential filename |

**BerkeleyGW:**

| Parameter | Description |
|---|---|
| `epsilon_cutoff` | Dielectric matrix cutoff (Ry) |
| `band_index_min/max` | Band range for QP corrections in sigma |
| `protected_cond_bands` | Parabands: protected conduction bands |
| `accumulation_window` | Parabands: accumulation window (Ry) |

**Structure generation:**

| Parameter | Default | Description |
|---|---|---|
| `Nreplicas` | 50 | Number of replicas |
| `seed` | 42 | Random seed |
| `delta` | 0.1 | Displacement std dev (Angstrom); 0.0 = no displacement |
| `eps_xx` | 0.0 | Strain xx component |
| `eps_yy` | 0.0 | Strain yy component |
| `eps_xy` | 0.0 | Strain xy (shear) component |

---

## Usage

```bash
# Use defaults from parameters.py and ATOMS.dat
python generate_BGWQE_inputs.py

# Override specific settings from the command line
python generate_BGWQE_inputs.py --nreplicas 10 --delta 0.05 --outdir test_run

# Apply 1% biaxial strain, no displacements
python generate_BGWQE_inputs.py --eps_xx 0.01 --eps_yy 0.01 --delta 0.0

# Use a different structure or parameter file
python generate_BGWQE_inputs.py --atoms WSe2.dat --params parameters_WSe2.py
```

All command-line flags override the corresponding value in `parameters.py`.

---

## After generating inputs

Run the SLURM job script from `examples/00_data_collection/job.sub` inside each
replica directory, then process the outputs with the preprocessing scripts:

```bash
# For each completed replica:
python ../pre_proc/map_orbitals_atoms.py \
    -projwfc_output replica_000/2-wfn_co/projwfc.out

python ../pre_proc/get_proj_for_graphs_and_eqp.py \
    -eqp replica_000/6-sigma/eqp1.dat \
    -Nval 13 \
    -proj_file replica_000/2-wfn_co/mos2.save/atomic_proj.xml \
    -orbital_mapping_file orbital_mapping.txt \
    -qe_input_file replica_000/1-scf/scf.in \
    -output replica_000/data.h5
```
