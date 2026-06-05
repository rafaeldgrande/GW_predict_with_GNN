# parameters.py
# All QE, BGW, and structure generation parameters for generate_BGWQE_inputs.py

# ── Quantum ESPRESSO ──────────────────────────────────────────────────────────
PREFIX       = 'mos2'
E_cutoff     = 35           # plane-wave cutoff (Ry)
Nk_co        = 6            # k-grid: Nk_co x Nk_co x 1
nbnds        = 30           # number of bands (>= number of electrons = 26)
conv_thr     = 1e-8         # SCF convergence threshold
pseudodir    = '/home/rrodriguesdelgrand/scratch/pseudos/nc-sr-04_pw_standard/'
pseudopotentials = {
    "Mo": "Mo.upf",
    "S":  "S.upf",
}

# ── BerkeleyGW ────────────────────────────────────────────────────────────────
epsilon_cutoff       = 10.0   # dielectric matrix cutoff (Ry)
band_index_min       = 9      # first band for QP corrections in sigma
band_index_max       = 20     # last  band for QP corrections in sigma
protected_cond_bands = 40     # parabands: number of protected conduction bands
accumulation_window  = 0.02   # parabands: accumulation window (Ry)

# ── Structure generation ──────────────────────────────────────────────────────
Nreplicas = 50     # number of displaced/strained replicas to generate
seed      = 42     # random seed for reproducibility

# Random atomic displacements (Gaussian, applied independently per atom and direction)
delta = 0.1        # standard deviation (Angstrom); set to 0.0 to disable

# Uniform strain (dimensionless; applied to lattice vectors and atomic positions)
eps_xx = 0.0       # xx component of the strain tensor
eps_yy = 0.0       # yy component
eps_xy = 0.0       # xy (shear) component; applied as eps_xy/2 off-diagonal
