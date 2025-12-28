import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from scipy.interpolate import RegularGridInterpolator
from matplotlib.path import Path

def parse_bxsf(filename):
    """
    Robustly parse .bxsf file, skipping comments and handling headers correctly.
    Returns Fermi energy, grid info, vectors, and band data.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    iterator = iter(lines)
    fermi_energy = None
    
    # 1. Robust Header Parsing
    while True:
        try:
            line = next(iterator).strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            if "Fermi Energy:" in line:
                fermi_energy = float(line.split()[-1])
            if "BANDGRID_3D_BANDS" in line:
                break
        except StopIteration:
            return None

    # 2. Parse Grid Information
    try:
        # Helper function to get the next N valid tokens
        def get_tokens(num):
            tokens = []
            while len(tokens) < num:
                line = next(iterator).strip()
                if not line or line.startswith('#') or "BANDGRID_3D_BANDS" in line:
                    continue
                tokens.extend(line.split())
            return tokens

        # Read number of bands
        token_nb = get_tokens(1)
        num_bands = int(token_nb[0])
        
        # Read grid dimensions
        tokens_grid = get_tokens(3)
        nx, ny, nz = map(int, tokens_grid)
        
        # Read origin
        tokens_org = get_tokens(3)
        origin = np.array(list(map(float, tokens_org)))
        
        # Read reciprocal lattice vectors
        tokens_v = get_tokens(9)
        v1 = np.array(list(map(float, tokens_v[0:3])))
        v2 = np.array(list(map(float, tokens_v[3:6])))
        v3 = np.array(list(map(float, tokens_v[6:9])))
        vectors = np.array([v1, v2, v3])
        
    except StopIteration:
        return None

    # 3. Parse Band Data
    raw_values = []
    for line in iterator:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # Skip 'BAND:' labels
        if "BAND:" in line:
            continue
        if "END_" in line:
            break
        
        # Parse numbers from the line
        parts = line.split()
        for p in parts:
            try:
                raw_values.append(float(p))
            except ValueError:
                pass
                
    raw_values = np.array(raw_values)
    total_points = nx * ny * nz
    
    bands_data = []
    if len(raw_values) == num_bands * total_points:
        for i in range(num_bands):
            start = i * total_points
            end = (i + 1) * total_points
            # Use correct C-order reshape (matches file stream order)
            band_grid = raw_values[start:end].reshape((nx, ny, nz))
            bands_data.append(band_grid)
            
    return {
        'Ef': fermi_energy,
        'nx': nx, 'ny': ny, 'nz': nz,
        'vectors': vectors,
        'bands': bands_data
    }

def get_bz_boundary(v1, v2):
    """
    Compute the 2D First Brillouin Zone (Voronoi Wigner-Seitz cell).
    """
    points = []
    # Generate 5x5 reciprocal lattice points around origin
    for i in range(-2, 3):
        for j in range(-2, 3):
            points.append(i * v1 + j * v2)
    points = np.array(points)
    
    # Construct Voronoi diagram
    vor = Voronoi(points)
    
    # Find the region corresponding to the origin (0,0)
    center_idx = np.argmin(np.linalg.norm(points, axis=1))
    region_idx = vor.point_region[center_idx]
    verts = vor.vertices[vor.regions[region_idx]]
    
    # Sort vertices by angle to draw a closed polygon
    centroid = np.mean(verts, axis=0)
    angles = np.arctan2(verts[:,1] - centroid[1], verts[:,0] - centroid[0])
    sort_idxs = np.argsort(angles)
    return verts[sort_idxs]

def plot_fermi_numerical_smooth(filename):
    data = parse_bxsf(filename)
    if not data:
        print("Error: Failed to parse data")
        return

    Ef = data['Ef']
    nx, ny = data['nx'], data['ny']
    bands = data['bands']
    
    v1 = data['vectors'][0, :2]
    v2 = data['vectors'][1, :2]
    
    # --- 1. Compute First Brillouin Zone Boundary ---
    bz_verts = get_bz_boundary(v1, v2)
    bz_path = Path(bz_verts)
    
    # --- 2. Prepare High-Resolution Plotting Grid ---
    # Define plotting range based on BZ bounding box
    min_x, max_x = bz_verts[:, 0].min(), bz_verts[:, 0].max()
    min_y, max_y = bz_verts[:, 1].min(), bz_verts[:, 1].max()
    padding = 0.1
    
    resolution = 500 # 500x500 high-resolution grid for smoothness
    grid_x = np.linspace(min_x - padding, max_x + padding, resolution)
    grid_y = np.linspace(min_y - padding, max_y + padding, resolution)
    PX, PY = np.meshgrid(grid_x, grid_y)
    
    # --- 3. Coordinate Mapping (Cartesian -> Fractional) ---
    # Map each pixel on screen to original grid coordinates (u, v)
    matrix = np.column_stack((v1, v2))
    inv_matrix = np.linalg.inv(matrix)
    
    cart_coords = np.vstack((PX.flatten(), PY.flatten()))
    frac_coords = inv_matrix @ cart_coords
    
    # Exploit periodicity: wrap coordinates back to [0, 1] interval
    U_flat = frac_coords[0, :] % 1.0
    V_flat = frac_coords[1, :] % 1.0
    
    # --- 4. Prepare Interpolator Source Grid ---
    src_u = np.linspace(0, 1, nx)
    src_v = np.linspace(0, 1, ny)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    plotted = False
    
    for idx, band in enumerate(bands):
        # Extract k_z=0 slice
        slice_2d = band[:, :, 0]
        
        if slice_2d.min() < Ef < slice_2d.max():
            # Create interpolator: use 'cubic' (cubic spline) for smooth interpolation
            interp = RegularGridInterpolator((src_u, src_v), slice_2d, 
                                           method='cubic', 
                                           bounds_error=False, fill_value=None)
            
            # Sample interpolation on high-resolution grid
            sample_points = np.column_stack((U_flat, V_flat))
            interp_vals = interp(sample_points).reshape(PX.shape)
            
            # Apply BZ mask (set values outside BZ to NaN)
            points_flat = np.column_stack((PX.flatten(), PY.flatten()))
            mask = bz_path.contains_points(points_flat).reshape(PX.shape)
            interp_vals[~mask] = np.nan
            
            # Plot contours
            ax.contour(PX, PY, interp_vals, levels=[Ef], 
                       colors=colors[idx % len(colors)], linewidths=2)
            plotted = True

    # Draw BZ boundary
    bz_plot = np.vstack((bz_verts, bz_verts[0]))
    ax.plot(bz_plot[:,0], bz_plot[:,1], 'k-', linewidth=1.5, label='First BZ')
    
    ax.set_aspect('equal')
    ax.set_title(f'NbSe2 Fermi Surface (Smoothed)\n$E_F = {Ef:.4f}$ eV')
    ax.set_xlabel(r'$k_x (\AA^{-1})$')
    ax.set_ylabel(r'$k_y (\AA^{-1})$')
    
    if not plotted:
        ax.text(0, 0, "No Fermi Surface found", ha='center')
        
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('fermi_surface_numerical_final.png', dpi=300)
    plt.show()

if __name__ == "__main__":

    plot_fermi_numerical_smooth('band.bxsf')
