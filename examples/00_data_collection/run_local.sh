#!/bin/bash
# Run DFT + GW data collection locally (small test only — use job.sub on a cluster)
# Assumes: Quantum ESPRESSO and BerkeleyGW are in PATH
# Directory structure expected:
#   1-scf/    contains scf.in, pw2bgw-1scf.in, projwfc.in
#   2-wfn_co/ contains bands.in, pw2bgw-2wfn.in
#   3-wfnq_co/contains bands.in, pw2bgw-2wfn.in (shifted k-grid)
#   4-parabands/ contains parabands.inp
#   5-epsilon/contains epsilon.inp
#   6-sigma/  contains sigma.inp

set -e

# Step 1: SCF
cd 1-scf/
pw.x < scf.in > scf.out
pw2bgw.x < pw2bgw-1scf.in > pw2bgw.out
echo "Step 1 (SCF) done"
cd ../

# Step 2: WFN
cd 2-wfn_co/
mkdir -p mos2.save/
cp ../1-scf/mos2.save/data-file-schema.xml mos2.save/
cp ../1-scf/mos2.save/charge-density* mos2.save/
pw.x < bands.in > bands.out
pw2bgw.x < pw2bgw-2wfn.in > pw2bgw.out
echo "Step 2 (WFN) done"
cd ../

# Step 3: WFNq
cd 3-wfnq_co/
mkdir -p mos2.save/
cp ../1-scf/mos2.save/data-file-schema.xml mos2.save/
cp ../1-scf/mos2.save/charge-density* mos2.save/
pw.x < bands.in > bands.out
pw2bgw.x < pw2bgw-2wfn.in > pw2bgw.out
wfn2hdf.x BIN wfn.complex WFN.h5
echo "Step 3 (WFNq) done"
cd ../

# Step 4: Parabands
cd 4-parabands/
ln -sf ../2-wfn_co/wfn.complex .
ln -sf ../2-wfn_co/VKB .
ln -sf ../2-wfn_co/VSC .
parabands.cplx.x > parabands.out
echo "Step 4 (Parabands) done"
cd ../

# Step 5: Epsilon
cd 5-epsilon/
ln -sf ../2-wfn_co/WFN.h5 .
ln -sf ../3-wfnq_co/WFN.h5 WFNq.h5
epsilon.cplx.x > epsilon.out
echo "Step 5 (Epsilon) done"
cd ../

# Step 6: Sigma → eqp.dat
cd 6-sigma/
ln -sf ../2-wfn_co/WFN.h5 WFN_inner.h5
ln -sf ../2-wfn_co/WFN.h5 WFN_outer.h5
ln -sf ../5-epsilon/eps0mat.h5 .
ln -sf ../5-epsilon/epsmat.h5 .
ln -sf ../2-wfn_co/VXC .
ln -sf ../2-wfn_co/RHO .
sigma.cplx.x > sigma.out
echo "Step 6 (Sigma) done — eqp.dat ready"
cd ../

# Step 7: Projwfc → atomic_proj.xml, projwfc.out
cd 2-wfn_co/
projwfc.x < projwfc.in > projwfc.out
echo "Step 7 (projwfc) done — atomic_proj.xml and projwfc.out ready"
cd ../
