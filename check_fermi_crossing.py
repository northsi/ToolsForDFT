import xml.etree.ElementTree as ET
import sys
import os

def check_bands(filename='vasprun.xml'):
    """
    Parses vasprun.xml to identify bands that cross the Fermi energy.
    """
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        return

    print(f"Reading {filename} ...")
    try:
        tree = ET.parse(filename)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return

    # 1. Get Fermi Energy
    efermi = None
    # Fermi energy is typically found in <calculation> -> <dos> -> <i name="efermi">
    # or we can search for any <i name="efermi"> tag.
    for i in root.iter('i'):
        if i.get('name') == 'efermi':
            try:
                efermi = float(i.text.strip())
                # Usually we want the last occurrence if multiple exist (e.g. from relaxation steps)
            except:
                continue
    
    if efermi is None:
        print("Error: Fermi energy (efermi) not found in file.")
        return
        
    print(f"Fermi Energy: {efermi:.4f} eV")

    # 2. Get Eigenvalues
    # The path is usually: <calculation> -> <eigenvalues> -> <array> -> <set> (spins)
    # We look for the <eigenvalues> node.
    
    eigen_node = root.find(".//calculation/eigenvalues/array/set")
    if eigen_node is None:
        # Try a more general search if the specific path fails
        eigen_node = root.find(".//eigenvalues/array/set")
        
    if eigen_node is None:
        print("Error: Eigenvalue data not found.")
        return

    # 3. Iterate over spin components (Spin 1 and potentially Spin 2)
    spins = eigen_node.findall('set')
    
    for s_idx, spin_set in enumerate(spins):
        spin_label = f"Spin {s_idx + 1}"
        if 'comment' in spin_set.attrib:
            spin_label += f" ({spin_set.attrib['comment']})"
            
        print(f"\n--- {spin_label} ---")
        
        # Store min and max eigenvalues for each band
        # key: band_index (1-based), value: [min_eigenval, max_eigenval]
        bands_stats = {}
        
        kpoints = spin_set.findall('set') # Each <set> inside spin corresponds to a k-point
        
        for kp in kpoints:
            # Each <r> tag is a band: energy occupation
            for b_idx, r in enumerate(kp.findall('r')):
                val = float(r.text.split()[0]) # The first column is energy
                band_idx = b_idx + 1
                
                if band_idx not in bands_stats:
                    bands_stats[band_idx] = [val, val]
                else:
                    if val < bands_stats[band_idx][0]:
                        bands_stats[band_idx][0] = val
                    if val > bands_stats[band_idx][1]:
                        bands_stats[band_idx][1] = val
        
        # Check for Fermi level crossing
        crossing_bands = []
        for b_idx in sorted(bands_stats.keys()):
            emin, emax = bands_stats[b_idx]
            
            # If the band spans across the Fermi energy
            if emin < efermi < emax:
                crossing_bands.append(b_idx)
                print(f"  Band {b_idx:3d}: Min = {emin:8.4f}, Max = {emax:8.4f}  <-- CROSSING")
            elif emax < efermi:
                # Fully occupied band (below Ef)
                pass
            elif emin > efermi:
                # Fully empty band (above Ef)
                pass
                
        if not crossing_bands:
            print("  No bands cross the Fermi level (System might be gapped).")
        else:
            print(f"  Indices of bands crossing Fermi level: {crossing_bands}")

if __name__ == "__main__":
    check_bands()
