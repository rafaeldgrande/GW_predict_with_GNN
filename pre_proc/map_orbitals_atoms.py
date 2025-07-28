import argparse

'''Usage:

python map_orbitals_atoms.py -projwfc_output projwfc.out
'''


def parse_wfcproj_output(filename):
    wfc_info = []  # List of tuples: (index, species, l, m)
    
    with open(filename, 'r') as f:
        for line in f:
            line_split = line.split()
            if len(line_split) > 0:
                if line_split[0] == 'state':
                    parts = line.strip().split()
                    # state #   1: atom   1 (Mo ), wfc  1 (l=0 m= 1)
                    # parts:  ['state', '#', '18:', 'atom', '3', '(S', '),', 'wfc', '2', '(l=1', 'm=', '3)']
                    # print('parts: ', parts)
                    idx = int(parts[2].strip(':')) - 1  # 0-based index
                    # species = parts[6].strip('()')  # e.g., "Mo" or "S"
                    species = parts[5].strip('()')  # e.g., "Mo" or "S"
                    # l = int(parts[9].split('=')[1])
                    # m = int(parts[10].split('=')[1])
                    s = int(parts[-4])
                    l = int(parts[-3].split('=')[1])
                    m = int(parts[-1].split(')')[0])
                    atom_index = int(parts[4])-1
                    
                    wfc_info.append((idx, species, s, l, m, atom_index))
    
    return wfc_info

def build_irrep_mapping(wfc_info):
    irrep_map = {}
    unique_orbitals = {}
    reduced_index = 0
    atoms_indexes = []

    for idx, species, s, l, m, atom_index in wfc_info:
        key = (species, s, l, m)  # Include s (principal quantum number)
        if key not in unique_orbitals:
            unique_orbitals[key] = reduced_index
            reduced_index += 1
        irrep_map[idx] = unique_orbitals[key]
        atoms_indexes.append(atom_index)
        
    print('unique_orbitals: ')
    print('atom_type s l m')
    for orbital in unique_orbitals:
        print(orbital)
        
    print("Number of orbitals:", len(wfc_info))
    print("Number of irreducible orbitals:", len(unique_orbitals))

    return irrep_map, atoms_indexes

def write_mapping(irrep_map, atoms_indexes, output_file='orbital_mapping.txt'):
    with open(output_file, 'w') as f:
        f.write('# atom_index original_orbital_index reduced_orbital_index\n')
        for orig_idx in sorted(irrep_map):
            atom_idx = atoms_indexes[orig_idx]
            reduced_idx = irrep_map[orig_idx]
            f.write(f"{atom_idx} {orig_idx} {reduced_idx}\n")
    print(f"Mapping written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-projwfc_output", "--projwfc_out_file", 
                        help="Output file from wfcproj.x", 
                        default="projwfc.out", type=str)
    args = parser.parse_args()

    # Step 1: Parse the wavefunction projections
    wfc_info = parse_wfcproj_output(args.projwfc_out_file)

    # Step 2: Build mapping from full orbital index to irreducible index
    irrep_map, atoms_indexes = build_irrep_mapping(wfc_info)

    # Step 3: Write the map to file
    write_mapping(irrep_map, atoms_indexes)

    print("Done!")