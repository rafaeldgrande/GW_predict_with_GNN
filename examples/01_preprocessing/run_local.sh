#!/bin/bash
# Preprocessing: convert QE + BGW outputs into an HDF5 dataset for the GNN
# Run from the directory that contains atomic_proj.xml, projwfc.out, eqp1.dat, qe.in

set -e

PREDIR="../../pre_proc"

# Step 1: Build orbital mapping from projwfc output
python $PREDIR/map_orbitals_atoms.py \
    -projwfc_output projwfc.out
# Output: orbital_mapping.txt

# Step 2: Build HDF5 dataset
#   -Nval: band index of top valence band (check your eqp1.dat)
python $PREDIR/get_proj_for_graphs_and_eqp.py \
    -eqp eqp1.dat \
    -Nval 13 \
    -proj_file atomic_proj.xml \
    -orbital_mapping_file orbital_mapping.txt \
    -qe_input_file qe.in \
    -output data.h5

echo "Done — output: data.h5"
echo "Add the path to data.h5 to your data_list.txt"
