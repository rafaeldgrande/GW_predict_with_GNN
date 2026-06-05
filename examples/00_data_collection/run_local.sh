#!/bin/bash
# Run DFT + GW data collection locally (for testing only — use the SLURM script on a cluster)

set -e

module load quantum-espresso  # adjust for your system

# Step 1: SCF
pw.x < scf.in > scf.out
echo "SCF done"

# Step 2: Bands (non-self-consistent on target k-grid)
pw.x < bands.in > bands.out
echo "Bands done"

# Step 3: Wavefunction projections
projwfc.x < projwfc.in > projwfc.out
echo "Projections done — outputs: atomic_proj.xml, projwfc.out"

# Step 4: BerkeleyGW — dielectric matrix
epsilon.x < epsilon.inp > epsilon.out
echo "Epsilon done"

# Step 5: BerkeleyGW — self-energy
sigma.x < sigma.inp > sigma.out
echo "Sigma done — outputs: eqp.dat"
