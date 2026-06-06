# GW Quasiparticle Corrections with Graph Neural Networks

A machine-learning framework for predicting GW quasiparticle (QP) energy corrections from DFT calculations using Graph Neural Networks (GNNs).

---

## Table of Contents

1. [Scientific Background](#scientific-background)
2. [Method Overview](#method-overview)
3. [Graph Representation](#graph-representation)
4. [GNN Architecture](#gnn-architecture)
5. [Project Structure](#project-structure)
6. [Workflow](#workflow)
7. [Usage](#usage)
8. [Dependencies](#dependencies)

---

## Scientific Background

### DFT and the Band Gap Problem

Density Functional Theory (DFT) provides an efficient way to compute the electronic structure of materials, but it systematically underestimates band gaps. This is because DFT eigenvalues are Kohn-Sham single-particle energies and lack the proper many-body exchange-correlation physics needed to describe excited electronic states.

### The GW Approximation

The GW approximation is a many-body perturbation theory (MBPT) method that corrects DFT eigenvalues by computing the electron self-energy $\Sigma$. The quasiparticle (QP) energies satisfy:

$$E_{nk}^{QP} = E_{nk}^{DFT} + \langle \psi_{nk} | \Sigma(E_{nk}^{QP}) - V_{xc} | \psi_{nk} \rangle$$

where $V_{xc}$ is the DFT exchange-correlation potential and $\Sigma = iGW$ is the self-energy expressed in terms of the Green's function $G$ and the screened Coulomb interaction $W$.

In practice, the QP correction is defined as:

$$\Delta E_{nk} = E_{nk}^{QP} - E_{nk}^{DFT}$$

GW calculations (e.g., via BerkeleyGW) are computationally expensive — scaling roughly as $O(N^3)$ or worse — making them prohibitive for large-scale screening or molecular dynamics applications.

### Goal

This code trains a GNN to predict $\Delta E_{nk}$ directly from DFT inputs, bypassing the expensive GW calculation while retaining its accuracy.

---

## Method Overview

The central idea is to represent each electronic state $(n, k)$ as a **molecular graph** where:

- Each **node** is an atom in the crystal
- Node features encode the **wavefunction character** of state $(n, k)$ at that atom via atomic orbital projections
- **Edges** connect neighboring atoms within a cutoff radius, with interatomic distances and bond angles as edge features

The GNN learns a mapping:

$$\{\text{orbital projections}_{nk}, E_{nk}^{DFT}, \text{crystal geometry}\} \;\longrightarrow\; \Delta E_{nk}$$

### Wavefunction Projections as Node Features

Each DFT wavefunction $|\psi_{nk}\rangle$ is projected onto a basis of localized atomic orbitals $|\phi_{Ro}\rangle$:

$$p_{nk,Ro} = |\langle \phi_{Ro} | \psi_{nk} \rangle|^2$$

where $R$ is the atom index and $o$ is the orbital index (labeled by principal quantum number $s$, angular momentum $l$, and magnetic quantum number $m$). These projections are produced by the `projwfc.x` code from Quantum ESPRESSO.

The node feature vector for atom $R$ in state $(n, k)$ is then the vector of orbital projections $\{p_{nk,Ro}\}_{o}$, capturing how much of the wavefunction's weight resides at that atom in each orbital channel.

---

## Graph Representation

### Construction

For each crystal structure, a graph is built using periodic boundary conditions (PBC) via ASE's `neighbor_list`:

- **Cutoff radius**: 3.0 Å (nearest neighbors)
- **Node features** `x`: shape `(N_atoms, N_orbitals)` — orbital projections per atom
- **Edge index**: pairs $(i, j)$ for all atoms $i, j$ within the cutoff
- **Edge attributes**: interatomic distances $d_{ij}$, shape `(N_edges, 1)`
- **Angle index**: triplets $(i, j, k)$ where $j$ is the central atom
- **Angles**: bond angles $\theta_{ijk}$ at atom $j$, shape `(N_triplets, 1)`
- **`edft`**: DFT eigenvalue $E_{nk}^{DFT}$ (scalar, per sample)
- **`y`**: target QP correction $\Delta E_{nk}$ (scalar, per sample)

Each $(n, k)$ pair produces one graph (same topology, different node features).

---

## GNN Architecture

Two model architectures are implemented in [main/gnn_def.py](main/gnn_def.py):

### `AngleGNN` (primary model)

A custom message-passing network that explicitly incorporates both **interatomic distances** and **bond angles**.

#### `AngleGNNLayer`

Each layer performs two message-passing steps:

**Distance-based messages:**

$$\mathbf{m}_{ij} = \mathbf{W}(d_{ij}) \cdot \mathbf{h}_j$$

where $\mathbf{W}(d_{ij})$ is a weight matrix produced by a small MLP applied to the scalar distance $d_{ij}$:

```
edge_mlp: Linear(1→32) → ReLU → Linear(32→in_channels×out_channels)
```

Messages are aggregated by summing over all neighbors of each node.

**Angle-based messages:**

Bond angles at the central atom $j$ contribute via:

$$\mathbf{a}_{ijk} = \text{MLP}_{angle}(\theta_{ijk})$$

$$\mathbf{a}_j = \sum_{i,k} \mathbf{a}_{ijk}$$

The combined aggregated features are: $\mathbf{h}_j' = \mathbf{m}_j + \mathbf{a}_j$

#### Full `AngleGNN` forward pass

```
Input: node features x (N_atoms, input_dim)
For each GNN layer:
    h = AngleGNNLayer(h, edge_index, edge_attr, angle_index, angles)
    h = BatchNorm1d(h)
    h = activation(h)     # ReLU by default
    h = Dropout(h)
h_graph = GlobalMeanPool(h)                    # (batch_size, hidden_dim_last)
h_cat   = concat([h_graph, Edft])              # (batch_size, hidden_dim_last + 1)
output  = MLP(h_cat)                           # (batch_size, 1)  →  ΔE_nk
```

The DFT energy $E_{nk}^{DFT}$ is concatenated after pooling so the model can learn a direct correction on top of the DFT result.

### `GNNOptunaModel` (distance-only baseline)

Uses `NNConv` from PyTorch Geometric — edge-conditioned convolution with distance features only, no angle terms. Same pooling + MLP readout structure.

### Hyperparameter search space

All architectural hyperparameters are optimized by [Optuna](https://optuna.org) (Bayesian optimization with TPE sampler):

| Parameter | Description |
|-----------|-------------|
| `n_layers` | Number of GNN layers (2–4) |
| `hidden_dim_layer_i` | Width of each hidden layer independently (32–128, step 32) |
| `dropout_rate` | Dropout probability (0.1–0.5) |
| `mlp_hidden_dim` | Hidden size of the readout MLP (64–256) |
| `learning_rate` | Adam learning rate (log-uniform, 1e-3–1e-1) |

---

## Project Structure

```
GW_predict_with_GNN/
├── pre_proc/          # Preprocessing: DFT/GW output → HDF5 dataset
│   ├── README.md      # ← full flag reference for preprocessing scripts
│   ├── map_orbitals_atoms.py
│   └── get_proj_for_graphs_and_eqp.py
├── main/              # GNN training and prediction
│   ├── README.md      # ← full flag reference for all training/prediction scripts
│   ├── gnn_def.py
│   ├── utils.py
│   ├── optimize_hyperparameters.py
│   ├── train_model.py
│   └── predict.py
└── tests/             # Automated test suite
    ├── README.md      # ← how to run tests and what each test checks
    ├── data_for_tests.h5
    └── test_*.py
```

> **See the `README.md` in each subdirectory for the complete list of flags and options for every script.**

### HDF5 data file structure

Produced by `get_proj_for_graphs_and_eqp.py`:

| Dataset | Shape | Description |
|---------|-------|-------------|
| `atom_orb_projections` | `(Nk, Nb, Natoms, Norbs)` | Orbital projections $p_{nk,Ro}$ |
| `Edft` | `(Nb, Nk)` | DFT eigenvalues (eV) |
| `qp_corrections` | `(Nb, Nk)` | GW corrections $\Delta E_{nk}$ (eV) |
| `atomic_positions` | `(Natoms, 3)` | Cartesian positions (Å) |
| `lattice_vectors` | `(3, 3)` | Lattice matrix (Å) |
| `atomic_species` | `(Natoms,)` | Element symbols |

---

## Workflow

### Step 0: Run DFT + GW calculations

Perform:
1. SCF + bands calculation with Quantum ESPRESSO (`pw.x`)
2. Wavefunction projections with `projwfc.x` → `atomic_proj.xml` and `projwfc.out`
3. GW calculation with BerkeleyGW → `eqp1.dat` (full QP solution)

### Step 1: Build orbital mapping

Parse the `projwfc.out` file to map the full set of atomic wavefunctions to a reduced irreducible set (collapsing orbitals that are equivalent by symmetry):

```bash
python pre_proc/map_orbitals_atoms.py -projwfc_output projwfc.out
# Output: orbital_mapping.txt
```

### Step 2: Build HDF5 dataset

```bash
python pre_proc/get_proj_for_graphs_and_eqp.py \
    -eqp eqp1.dat \
    -Nval <valence_band_index> \
    -proj_file atomic_proj.xml \
    -orbital_mapping_file orbital_mapping.txt \
    -qe_input_file qe.in \
    -output data.h5
```

Create a `data_list.txt` file listing all HDF5 files (one per line):

```
/path/to/data.h5
```

### Step 3: Hyperparameter optimization

```bash
cd main/
python optimize_hyperparameters.py \
    --file_list_data ../data_list.txt \
    --use_angles --use_distances \
    --n_trials_Bayesian_optimization 50 \
    --total_epochs_trial 30 \
    --n_layers_min 2 --n_layers_max 4 \
    --hidden_dims_min 32 --hidden_dims_max 128 \
    --batch_size 32
# Output: best_params.json, optimization_history.html, param_importances.html
```

### Step 4: Train the model

```bash
python train_model.py \
    --file_list_data ../data_list.txt \
    --model best_params.json \
    --epochs_train_model 500 \
    --patience_training 50 \
    --batch_size 32 \
    --output_weights_file best_model.pth \
    --plot_data
# Output: best_model.pth, loss_mae.png, pred_vs_true_qp.png
```

### Step 5: Predict on new data

```bash
python predict.py \
    --file_list_data new_data_list.txt \
    --model best_params.json \
    --weights_file best_model.pth \
    --predictions_file predictions.npz \
    --plot_data
# Output: predictions.npz, eqp_from_GNN.dat
```

The file `eqp_from_GNN.dat` contains two columns: `Edft, Eqp_GNN` (eV), one row per state $(n, k)$.

---

## Usage

All scripts are run from the `main/` directory and accept `--help` for full argument descriptions:

```bash
python optimize_hyperparameters.py --help
python train_model.py --help
python predict.py --help
```

Training and prediction logs are written to `logs/` with timestamps.

---

## Dependencies

- Python 3.11
- PyTorch
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [ASE](https://wiki.fysik.dtu.dk/ase/) (Atomic Simulation Environment)
- [Optuna](https://optuna.org) (hyperparameter optimization)
- [MLflow](https://mlflow.org) (experiment tracking)
- NumPy, SciPy, h5py, matplotlib, plotly

### Installation

The recommended way is to create the conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate gnn_gw_env
```

`environment.yml` pins all package versions for reproducibility. It contains
three commented PyTorch install lines — the NVIDIA GPU (CUDA 12.x) line is
active by default. On a CPU-only machine or on macOS (Apple Silicon / MPS),
edit the file to comment out the GPU line and uncomment the appropriate one
before creating the environment.

Alternatively, install manually into an existing environment:

```bash
pip install torch torch_geometric optuna mlflow h5py ase numpy scipy matplotlib plotly
```
