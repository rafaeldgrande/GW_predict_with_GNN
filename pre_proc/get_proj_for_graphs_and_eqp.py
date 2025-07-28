
import xml.etree.ElementTree as ET
from ase.io.espresso import read_espresso_in
import numpy as np
import argparse
import h5py

ry2eV = 13.605698066

'''

This code gets wavefunction data to be read by a Graph Neural Network

It reads the projections for the wavefunction psi_nk, where n is the band index and k is the k-point index

Each wavefunction is projected into a set of atomic orbitals

|psi_nk> = sum_Ro <phi_Ro|psi_nk> |phi_Ro>,

where R is the atomic index and o is the orbital index

The dataset will have shape (Nb, Nk, NR, No) -> bands, kpoints, atom, orbital
In addition to that, we also have the DFT energies QP corrections,
and atomic positions and lattice vectors (to be build graphs)

The file data.h5 will have structure:

data:
    Nb
    Nk
    NR
    No
    projections (Nk, Nb, NR, No) -> bands, kpoints, atom, orbital
    DFT energies (Nb, Nk)
    QP corrections (Nb, Nk)
    atomic positions (NR, 3)
    lattice vectors (3, 3)
'''

def read_eqp_dat_file(eqp_file, Nval=0):
    
    print('Reading file: ', eqp_file)
    
    bands_dft, bands_qp = [], []
    
    # loading file
    
    data = np.loadtxt(eqp_file)

    # first getting number of bands in this file. first line is:   0.000000000  0.000000000  0.000000000      13
    Nbnds = int(data[0, 3])
    
    # getting list of band indexes
    band_indexes_temp = data[1:Nbnds+1, 1] 
    band_indexes = [int(band_index) for band_index in band_indexes_temp]
    band_indexes_pythonic = [int(band_index-1) for band_index in band_indexes_temp]
        
    # now we get the k points in this file
    Kpoints = data[0::Nbnds+1] # get lines 0, Nbnds+1, 2*(Nbnds+2), ...
    Kpoints = Kpoints[:, :3] # remove last collumn with 
    
    Nk = len(Kpoints)
    print(f'Number of kpoints {Nk}')

    for ibnd in range(Nbnds):
        temp = data[ibnd+1::Nbnds+1]
        bands_dft.append(temp[:, 2])
        bands_qp.append(temp[:, 3])
        
    bands_qp = np.array(bands_qp)
    bands_dft = np.array(bands_dft)
        
    if Nval in band_indexes:
        index_nval = band_indexes.index(Nval)
        bands_dft -= np.max(bands_dft[index_nval]) # making the max(top valence band) = 0
        bands_qp -= np.max(bands_qp[index_nval])
    else:
        print(f"Did not find band {Nval} on eqp file {eqp_file}")
        
    # print('band indexes: ', band_indexes)
    # print('band indexes pythonic: ', band_indexes_pythonic)
    # print('Nval: ', Nval)
    # print('band_indexes.index(Nval): ', band_indexes.index(Nval))
                
    return bands_dft, bands_qp, Kpoints, Nk, band_indexes_pythonic

def get_orbital_mapping(orbital_mapping_file):
    print('Reading file: ', orbital_mapping_file) # atom_index original_orbital_index reduced_orbital_index
    data = np.loadtxt(orbital_mapping_file)
    print('data shape: ', data.shape)
           
    return data


def read_proj_from_file(proj_file, orbital_mapping):
    
    num_atoms = int(orbital_mapping[:, 0].max()) + 1 
    num_red_orbitals = int(orbital_mapping[:, 2].max()) + 1
    
    # reading projections 
    
    tree = ET.parse(proj_file)
    root = tree.getroot()
    
    header = root.find("HEADER")

    # Extract values
    num_bands = int(header.attrib["NUMBER_OF_BANDS"])
    num_kpoints = int(header.attrib["NUMBER_OF_K-POINTS"])
    num_atomic_wfc = int(header.attrib["NUMBER_OF_ATOMIC_WFC"])
    
    # Print results
    print(f'Reading projections from file: {proj_file}')
    print(f"Number of bands: {num_bands}")
    print(f"Number of k-points: {num_kpoints}")
    print(f"Number of atomic WFCs: {num_atomic_wfc}")
    
    eigenstates = root.find("EIGENSTATES")
    kpoints = eigenstates.findall("K-POINT")
    energies_tags = eigenstates.findall("E")
    projs_tags = eigenstates.findall("PROJS")

    print(f"Number of K-POINT tags: {len(kpoints)}")
    print(f"Number of E (energies) tags: {len(energies_tags)}")
    print(f"Number of PROJS tags: {len(projs_tags)}")
    
    # Parse projections: initialize 3D array [nk, n_atomic_wfc, nb]
    PROJECTIONS = np.zeros((num_kpoints, num_bands, num_atoms, num_red_orbitals))


    for ik in range(num_kpoints):
        projs_tag = projs_tags[ik]
        atomic_wfcs = projs_tag.findall("ATOMIC_WFC")
        if len(atomic_wfcs) != num_atomic_wfc:
            print(f"Warning: number of ATOMIC_WFC tags ({len(atomic_wfcs)}) != header value ({num_atomic_wfc}) at k-point {ik}")
        for iwfc, atomic_wfc in enumerate(atomic_wfcs):
            iatom = int(orbital_mapping[iwfc, 0])
            iwfc_red = int(orbital_mapping[iwfc, 2])
            # print('iatom: ', iatom, 'type(iatom): ', type(iatom))
            # print('iwfc_red: ', iwfc_red, 'type(iwfc_red): ', type(iwfc_red))
            proj_text = atomic_wfc.text.strip()
            proj_values = [float(x) for x in proj_text.split()]
            expected_len = 2 * num_bands  # because real + imag per band
            if len(proj_values) != expected_len:
                print(f"Warning: number of projections ({len(proj_values)}) != expected ({expected_len}) at k-point {ik}, atomic_wfc {iwfc}")
            
            # Convert pairs of floats into complex numbers
            n = min(len(proj_values) // 2, num_bands)
            proj_complex = [complex(proj_values[2*i], proj_values[2*i + 1]) for i in range(n)]
            
            # Store squared magnitude |projection|^2
            PROJECTIONS[ik, :n, iatom, iwfc_red] = [abs(cval)**2 for cval in proj_complex]

    print('Projections shape:', PROJECTIONS.shape, '= (nk, n_bands, n_atoms, n_red_orbitals)')

    # Parse energies as before
    ENERGIES = np.zeros((num_bands, num_kpoints))
    for ik in range(num_kpoints):
        energies_text = energies_tags[ik].text.strip()
        energies = [float(e) for e in energies_text.split()]
        for ib, energy in enumerate(energies):
            ENERGIES[ib, ik] = energy
    ENERGIES = np.array(ENERGIES) * ry2eV # shape (nb, nk)
    print('Energies shape in projections file: ', ENERGIES.shape, '= (nb, nk)')

    return PROJECTIONS, num_atomic_wfc, num_bands, num_kpoints, ENERGIES

def read_qe_input(filename):
    """
    Reads Quantum ESPRESSO input file using ASE and returns:
    - atomic positions in Cartesian coordinates [Å]
    - lattice vectors (3x3 matrix)
    - atomic species (list of strings)
    """
    
    print('Reading QE input file: ', filename)
    
    atoms = read_espresso_in(filename)

    positions = atoms.get_positions()                # shape (N_atoms, 3)
    lattice = atoms.get_cell().array                 # shape (3, 3)
    atomic_species = atoms.get_chemical_symbols()    # list of strings, e.g., ['Mo', 'S', 'S']
    
    print(f'Found {len(positions)} atoms')
    print('Lattice vectors: ', lattice)

    return positions, lattice, atomic_species

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-eqp", "--eqp_file", help="eqp.dat file from BGW ", default='eqp.dat')
    parser.add_argument("-Nval", "--Nval", help="Number of valence band", default=0)
    parser.add_argument("-proj_file", "--proj_file", help="File with projected states. Usually celled atomic_proj.xml. Produced by projwfc.x from Quantum espresso", default='atomic_proj.xml')
    parser.add_argument("-output", "--output_file", help="Output file name. It is a h5 file.", default='data.h5')
    parser.add_argument("-plot_data", "--plot_data", help="Plot data. True or False", default=False)
    parser.add_argument("-orbital_mapping_file", "--orbital_mapping_file", help="File with orbital mapping. Produced by summarize_atoms.py", default='orbital_mapping.txt')
    parser.add_argument("-dont_use_eqp", "--dont_use_eqp", help="Dont use eqp.dat file and consider qp corrections to be zero. DFT energies will be loaded from proj_file", default=False)
    parser.add_argument("-qe_input_file", "--qe_input_file", help="Quantum espresso input file. We get the atomic positions and lattice vectors from it", default='qe.in')
    
    args = parser.parse_args()
    eqp_file = args.eqp_file
    Nval = int(args.Nval)
    proj_file = args.proj_file
    output_file = args.output_file
    plot_data = bool(args.plot_data)
    orbital_mapping_file = args.orbital_mapping_file
    dont_use_eqp = bool(args.dont_use_eqp)
    qe_input_file = args.qe_input_file
    
    # reading QE input file
    positions, lattice, atomic_species = read_qe_input(qe_input_file)
    Natoms = len(positions)
    
    # reading orbital mapping
    orbital_mapping = get_orbital_mapping(orbital_mapping_file)
    # n_atomic_wfc_reduced = max(orbital_mapping)+1
    
    PROJECTIONS, num_atomic_wfc, num_bands, num_kpoints, ENERGIES = read_proj_from_file(proj_file, orbital_mapping)
    
    if dont_use_eqp:
        Edft = ENERGIES
        Edft -= np.max(ENERGIES[Nval-1, :]) # setting max(Ev(Nval)) = 0. All states included in this case!
        qp_corrections = np.zeros(ENERGIES.shape)
    else:
        # getting dft and qp energy levels
        Edft, Eqp, kpoints_sigma, Nk, bands_indexes_sigma = read_eqp_dat_file(eqp_file, Nval=Nval)
        qp_corrections = Eqp - Edft # eV, shape (Nbnds, Nk)
        # shape of Edft, Eqp and qp_corrections: (Nbnds, Nk)
        PROJECTIONS = PROJECTIONS[:, bands_indexes_sigma, :, :]

    
    # # just checking if everything is ok
    # DATA_LABELS = ['atomic_positions', 'lattice_vectors', 'atom_orb_projections', 'Edft', 'qp_corrections']
    # DATA_EXPORT = [positions, lattice, PROJECTIONS, Edft, qp_corrections]
    
    # for i in range(len(DATA_LABELS)):
    #     print(DATA_LABELS[i], DATA_EXPORT[i].shape)
    
    
    # save data in h5 file
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('atom_orb_projections', data=PROJECTIONS) # shape (nk, n_bands, n_atoms, n_red_orbitals)
        f.create_dataset('Edft', data=Edft)                       # shape (Nbnds, Nk)
        f.create_dataset('qp_corrections', data=qp_corrections)   # shape (Nbnds, Nk)
        f.create_dataset('atomic_positions', data=positions)  # shape (num_atoms, 3)
        f.create_dataset('lattice_vectors', data=lattice)     # shape (3, 3)
        f.create_dataset('atomic_species', data=atomic_species, dtype=h5py.string_dtype())

    if plot_data:
        print('Plotting data...')
        import matplotlib.pyplot as plt
        
        # f, axs = plt.subplots(DATA_EXPORT.shape[1], DATA_EXPORT.shape[1], figsize=(20, 20))
        # for i in range(DATA_EXPORT.shape[1]):
        #     for j in range(DATA_EXPORT.shape[1]):
        #         axs[i, j].scatter(DATA_EXPORT[:, i], DATA_EXPORT[:, j])
        
        #     axs[0, i].set_xlabel(f'Column {i}')
        #     axs[i, 0].set_ylabel(f'Column {i}')
        # f.savefig('scatterplot.png')

        plt.figure(figsize=(6, 6))
        plt.scatter(Edft.flatten(), qp_corrections.flatten())
        plt.xlabel('Edft (eV)')
        plt.ylabel('Eqp - Edft (eV)')
        plt.grid()
        plt.savefig('Edft_vs_Eqp.png')

