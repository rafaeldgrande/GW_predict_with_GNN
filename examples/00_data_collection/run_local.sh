#!/bin/bash
# Run DFT + GW data collection locally (small test — use job.sub on a cluster)
set -e

# Step 1: SCF
cd 1-scf/
pw.x < scf.in > scf.out
pw2bgw.x < pw2bgw.in > pw2bgw.out
echo "Step 1 (SCF) done"
cd ../

# Step 2: WFN + projwfc
cd 2-wfn_co/
mkdir -p mos2.save/
cp ../1-scf/mos2.save/data-file-schema.xml mos2.save/
cp ../1-scf/mos2.save/charge-density* mos2.save/
pw.x < bands.in > bands.out
pw2bgw.x < pw2bgw.in > pw2bgw.out
projwfc.x < projwfc.in > projwfc.out
echo "Step 2 (WFN + projwfc) done"
cd ../

# Step 3: WFNq (shifted k-grid)
cd 3-wfnq_co/
mkdir -p mos2.save/
cp ../1-scf/mos2.save/data-file-schema.xml mos2.save/
cp ../1-scf/mos2.save/charge-density* mos2.save/
pw.x < bands.in > bands.out
pw2bgw.x < pw2bgw.in > pw2bgw.out
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

# Step 6: Sigma
cd 6-sigma/
ln -sf ../2-wfn_co/WFN.h5 WFN_inner.h5
ln -sf ../2-wfn_co/WFN.h5 WFN_outer.h5
ln -sf ../5-epsilon/eps0mat.h5 .
ln -sf ../5-epsilon/epsmat.h5 .
ln -sf ../2-wfn_co/VXC .
ln -sf ../2-wfn_co/RHO .
sigma.cplx.x > sigma.out
echo "Step 6 (Sigma) done"
cd ../

echo ""
echo "Outputs ready for GNN preprocessing:"
echo "  2-wfn_co/mos2.save/atomic_proj.xml"
echo "  2-wfn_co/projwfc.out"
echo "  6-sigma/eqp.dat"
