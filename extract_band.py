import numpy as np
import os
import sys

def get_fermi_from_doscar(filename='DOSCAR'):
    """
    Reads the Fermi energy from the DOSCAR file.
    The Fermi energy is the 4th number on the 6th line.
    """
    if not os.path.exists(filename):
        print(f"Warning: '{filename}' not found. Cannot read Fermi energy.")
        return None

    try:
        with open(filename, 'r') as f:
            # Read line 6 (index 5)
            # Lines 0-4 are header, Line 5 contains Emax, Emin, NEDOS, Efermi, weight
            lines = f.readlines()
            if len(lines) < 6:
                print("Error: DOSCAR file is too short.")
                return None
            
            line6 = lines[5].strip()
            parts = line6.split()
            
            # Efermi is usually the 4th element (index 3)
            # Standard DOSCAR line 6: <EMAX> <EMIN> <NEDOS> <EFERMI> <1.000>
            if len(parts) >= 4:
                efermi = float(parts[3])
                return efermi
            else:
                print("Error: Line 6 of DOSCAR does not have enough columns.")
                return None
                
    except Exception as e:
        print(f"Error reading DOSCAR: {e}")
        return None

def extract_bands_data(filename='EIGENVAL', target_bands=[]):
    """
    Parses the VASP EIGENVAL file to extract k-point coordinates and energy 
    for multiple band indices.
    """
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return None, None

    with open(filename, 'r') as f:
        lines = f.readlines()

    try:
        # Header parsing
        # Line 6 (index 5) typically contains: N_ions, N_dim, N_kpoints, N_bands
        header_data = lines[5].split()
        num_kpoints = int(header_data[1])
        num_bands = int(header_data[2])
        
        print(f"EIGENVAL Header -> K-points: {num_kpoints}, Bands: {num_bands}")
        
        # Validate requested bands
        valid_bands = []
        for b in target_bands:
            if 1 <= b <= num_bands:
                valid_bands.append(b)
            else:
                print(f"Warning: Band {b} is out of range [1, {num_bands}]. Skipping.")
        
        if not valid_bands:
            print("Error: No valid bands selected.")
            return None, None
            
    except (IndexError, ValueError):
        print("Error: Could not parse EIGENVAL header. The format might differ.")
        return None, None

    kpoints_list = []
    # Initialize a list for each band to store energies
    energies_map = {b: [] for b in valid_bands}

    # The data section usually starts after line 7 (index 6)
    current_line = 6
    
    # Process each k-point block
    for k_idx in range(num_kpoints):
        # 1. Find the k-point header line (skip empty lines if any)
        while current_line < len(lines):
            line = lines[current_line].strip()
            if not line: # skip empty lines
                current_line += 1
                continue
            break
        
        if current_line >= len(lines):
            break

        # 2. Parse k-point coordinates
        # Format: x_frac y_frac z_frac weight
        try:
            k_line_parts = lines[current_line].split()
            kx, ky, kz = map(float, k_line_parts[:3])
            kpoints_list.append([kx, ky, kz])
        except ValueError:
            print(f"Error parsing k-point at line {current_line+1}")
            return None, None

        # 3. Extract energies for each requested band
        for b in valid_bands:
            # The line for band 'b' is 'b' lines below the k-point header
            band_entry_line = current_line + b
            
            try:
                band_data = lines[band_entry_line].split()
                # Format: Band_Index Energy Occupation
                energy = float(band_data[1])
                energies_map[b].append(energy)

            except (IndexError, ValueError):
                 print(f"Error parsing energy for band {b} at k-point {k_idx+1}")
                 energies_map[b].append(np.nan)

        # Advance pointer to the next k-point block
        current_line += (num_bands + 1)

    # Convert lists to numpy arrays
    for b in valid_bands:
        energies_map[b] = np.array(energies_map[b])
        
    return np.array(kpoints_list), energies_map

# --- Main Execution ---
if __name__ == "__main__":
    eigenval_file = "EIGENVAL"
    doscar_file = "DOSCAR"
    
    # --- SETTINGS ---
    # List the bands you want to extract (1-based index)
    target_bands_list = [49, 50, 51, 52] 
    # ----------------
    
    # 1. Get Fermi Energy
    efermi = get_fermi_from_doscar(doscar_file)
    
    if efermi is None:
        print("Could not retrieve Fermi energy. Energies will NOT be shifted.")
        shift_energy = 0.0
    else:
        print(f"Fermi Energy found: {efermi:.6f} eV")
        print("Energies will be shifted by -E_fermi (i.e., E_fermi set to 0).")
        shift_energy = efermi

    # 2. Extract Data
    print(f"\nExtracting data for Bands {target_bands_list} from {eigenval_file}...")
    k_vecs, engs_dict = extract_bands_data(eigenval_file, target_bands_list)
    
    if k_vecs is not None:
        print("\nExtraction Successful!")
        print("-" * 70)
        
        # Create header string
        header_str = f"{'K-point':<8} {'Kx':<10} {'Ky':<10} {'Kz':<10}"
        for b in target_bands_list:
            header_str += f" {'E_B'+str(b)+'-Ef':<15}"
        print(header_str)
        print("-" * 70)
        
        # Apply Shift and Print Preview
        # We apply shift directly to the dictionary values
        for b in target_bands_list:
            engs_dict[b] = engs_dict[b] - shift_energy

        # Print preview (first 10 points)
        for i in range(min(10, len(k_vecs))):
            kp = k_vecs[i]
            row_str = f"{i+1:<8} {kp[0]:<10.6f} {kp[1]:<10.6f} {kp[2]:<10.6f}"
            for b in target_bands_list:
                val = engs_dict[b][i]
                row_str += f" {val:<15.6f}"
            print(row_str)
            
        # 3. Save to File
        output_file = "extracted_bands_shifted.dat"
        
        # Prepare data array for saving
        columns = [k_vecs[:, 0], k_vecs[:, 1], k_vecs[:, 2]]
        header_labels = "Kx Ky Kz"
        
        for b in target_bands_list:
            columns.append(engs_dict[b])
            header_labels += f" E_Band_{b}(eV-Ef)"
            
        data_to_save = np.column_stack(columns)
        
        # Add metadata to the header
        full_header = f"Fermi Energy used: {shift_energy:.6f} eV\n" + header_labels
        
        np.savetxt(output_file, data_to_save, 
                   header=full_header, 
                   fmt="%12.8f")
        print(f"\nData saved to '{output_file}'")
