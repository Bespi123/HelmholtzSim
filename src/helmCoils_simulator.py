# Import dependencies
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import Union
import src.plotMagneticField as hplot

# Define global constants
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space
rotz_180 = np.array([
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, 1]
])

class CoilParameters:
    def __init__(self, coils_number: int, length: Union[float, list, np.ndarray], 
                 distance: Union[float, list, np.ndarray], turns: Union[int, list, np.ndarray], 
                 current: float, rot_matrix: np.ndarray):
        """
        Initialize Helmholtz coil parameters.

        Args:
            coils_number (int): Number of Helmholtz coils.
            length (float | list | np.ndarray): Length(s) of the Helmholtz testbed side.
            distance (float | list | np.ndarray): Distance(s) between Helmholtz coils.
            turns (int | list | np.ndarray): Number of turns in each coil.
            current (float): Electric current applied to the coils.
            rot_matrix (np.ndarray): Rotation matrix for coordinate transformation.
        """
        # Convert inputs to NumPy arrays
        self.L = np.atleast_1d(length)  
        self.h = np.atleast_1d(distance)   
        self.N = np.atleast_1d(turns) if isinstance(turns, (list, np.ndarray)) else np.array([turns])  
        self.I = current
        self.A = rot_matrix

        # Other parameters
        self.coils_number = coils_number
        
        # Validate and expand `length`
        if self.L.shape[0] not in {1, coils_number}:
            raise ValueError(f"Invalid length size. Expected 1 or {coils_number}, got {self.L.shape[0]}")
        elif self.L.shape[0] == 1:
            self.L = self.L[0] * np.ones((coils_number,))

        # Validate and expand `turns`
        if self.N.shape[0] not in {1, coils_number}:
            raise ValueError(f"Invalid turns size. Expected 1 or {coils_number}, got {self.N.shape[0]}")
        elif self.N.shape[0] == 1:
            self.N = self.N[0] * np.ones((coils_number,))

        # Validate and expand `height`
        if self.h.shape[0] not in {1, coils_number - 1}:
            raise ValueError(f"Invalid height size. Expected 1 or {coils_number - 1}, got {self.h.shape[0]}")
        elif self.h.shape[0] == 1:
            if self.coils_number == 1:
                self.h = [0]
            else:
                self.h = self.h[0] * np.ones((coils_number - 1,))

        # Validate `rot_matrix`
        if self.A.shape != (3, 3):
            raise ValueError(f"Invalid rotation matrix shape. Expected (3,3), got {self.A.shape}")
        
        self.a = self.L / 2  # Half Helmholtz testbed length side
        self.pos =  self.get_spires_position()

    def update_parameters(self, coils_number: int = None, 
                        length: Union[float, list, np.ndarray] = None,
                        distance: Union[float, list, np.ndarray] = None,
                        turns: Union[int, list, np.ndarray] = None,
                        current: float = None,
                        rot_matrix: np.ndarray = None):
        """
        Update the coil parameters during optimization or any time.

        Args:
            coils_number (int, optional): New number of Helmholtz coils.
            length (float | list | np.ndarray, optional): New length(s) for the coils.
            distance (float | list | np.ndarray, optional): New distance(s) for the coils.
            turns (int | list | np.ndarray, optional): New number of turns for the coils.
            current (float, optional): New electric current applied to the coils.
            rot_matrix (np.ndarray, optional): New rotation matrix for coordinate transformation.
        """
        if coils_number is not None:
            self.coils_number = coils_number

        if length is not None:
            self.L = np.atleast_1d(length)
            if self.L.shape[0] not in {1, self.coils_number}: 
                raise ValueError(f"Invalid length size. Expected 1 or {self.coils_number}, got {self.L.shape[0]}")
            elif self.L.shape[0] == 1:
                self.L = self.L[0] * np.ones((self.coils_number,))

        if distance is not None:
            self.h = np.atleast_1d(distance)
            if self.h.shape[0] not in {1, self.coils_number - 1}:
                raise ValueError(f"Invalid height size. Expected 1 or {self.coils_number - 1}, got {self.h.shape[0]}")
            elif self.h.shape[0] == 1:
                self.h = self.h[0] * np.ones((self.coils_number - 1,))

        if turns is not None:
            self.N = np.atleast_1d(turns)
            if self.N.shape[0] not in {1, self.coils_number}: 
                raise ValueError(f"Invalid turns size. Expected 1 or {self.coils_number}, got {self.N.shape[0]}")
            elif self.N.shape[0] == 1:
                self.N = self.N[0] * np.ones((self.coils_number,))

        if current is not None:
            self.I = current

        if rot_matrix is not None:
            self.A = rot_matrix

        # Recalculate the coil positions after parameter update
        self.a = self.L / 2  # Half Helmholtz testbed length side
        self.pos = self.get_spires_position()


    def get_spires_position(self):
        """
        Compute the positions of the spires in the Helmholtz coil system.

        Returns:
            np.ndarray: Array with computed positions.
        """
        coils_number = self.coils_number  # Number of coils
        h = self.h  # Heights between coils

        # Initialize arrays with NaN values
        d = np.full(coils_number, np.nan)   # Temporary array for displacement values
        d1 = np.full(coils_number, np.nan)  # Final array for cumulative positions

        o = coils_number // 2  # Compute middle index (integer division)

        # Compute displacement values for each coil except the last one
        if self.coils_number == 1:
            d1 = [0]
        else:
            for j in range(coils_number - 1):
                if j < o - (coils_number % 2 == 0):  
                    # For coils before the middle point
                    d[j] = -h[j]  
                elif j == o - (coils_number % 2 == 0):  
                    # Middle coil case: If even, split height; if odd, set zero
                    d[j] = -h[j] / 2 if coils_number % 2 == 0 else 0
                    d[j + 1] = h[j] / 2 if coils_number % 2 == 0 else h[j]
                else:
                    # For coils after the middle point
                    d[j + 1] = h[j]              

            # Compute cumulative positions
            for j in range(coils_number):
                if j <= o - (coils_number % 2 == 0):  
                    # Sum displacements from current position to the middle point
                    d1[j] = np.sum(d[j:o - (coils_number % 2 == 0) + 1]) 
                else:
                    # Sum displacements from the middle point onward
                    d1[j] = np.sum(d[o - (coils_number % 2 == 0) + 1:j + 1]) 

        return d1  # Return computed coil positions

    def square_spires(self, num_seg, b=None):
        """
        Generates coordinates for multiple square or rectangular coils in 3D space, transformed by matrix A.

        Parameters:
        num_seg (int): Number of segments per side.
        b (float, optional): Half the vertical side length (along the Z-axis). If not provided, b = a (square).t

        Returns:
            list: List of arrays, each representing a coil shape (4, 3, num_seg).
        """
        # Use `b = a` if not provided
        b = np.atleast_1d(b) if b is not None else self.a

        spires = []

        for i in range(self.coils_number):
            L0_half = self.a[i]
            L1_half = b[i]

            # Generate evenly spaced points for the y and z coordinates along the square sides
            y_coords = np.linspace(L0_half, -L0_half, num_seg)
            z_coords = np.linspace(-L1_half, L1_half, num_seg)

            # Define the 3D coordinates of the four sides of a square coil
            spire = np.array([
                [np.zeros(num_seg), y_coords, L1_half * np.ones(num_seg)],   # Top edge
                [np.zeros(num_seg), -L0_half * np.ones(num_seg), -z_coords], # Right edge
                [np.zeros(num_seg), -y_coords, -L1_half * np.ones(num_seg)], # Bottom edge
                [np.zeros(num_seg), L0_half * np.ones(num_seg), z_coords]    # Left edge
            ])

            displacement = np.array([self.pos[i], 0, 0]).reshape(3, 1)

            # Transform the coordinates of the second coil using the matrix A and apply the displacement
            spire = np.einsum('ij,ljk->lik', self.A, spire - displacement[None,: , :])
            spires.append(spire)
            
        spires = np.array(spires)
        spires = spires.transpose(0, 2, 1, 3).reshape(self.coils_number, 3, -1)

        return spires

    def circular_spires(self, num_seg):
        """
        Generates coordinates of multiple circular coils (spirals) in 3D space.

        Parameters:
            num_seg (int): Number of segments for the full spiral.

        Returns:
            np.ndarray: Array of shape (num_coils, 3, 4*num_seg) containing the coils.
        """
        spires = []

        for i in range(self.coils_number):
            r = self.L[i] / 2  # Radio del círculo

            # Generar ángulos para una circunferencia completa (espiral continua)
            theta_vals = np.linspace(0, 2 * np.pi, 4 * num_seg, endpoint=False)  

            # Parametrización de la espiral
            x_coords = np.zeros(4 * num_seg)  # x permanece constante
            y_coords = r * np.sin(theta_vals)  
            z_coords = r * np.cos(theta_vals)  

            # Formar la matriz de coordenadas con forma (3, 4*num_seg)
            spire = np.array([x_coords, y_coords, z_coords])

            displacement = np.array([self.pos[i], 0, 0]).reshape(3, 1)

            # Aplicar transformación con matriz A y desplazar
            spire = np.einsum('ij,jk->ik', self.A, spire - displacement)

            spires.append(spire)

        return np.array(spires)  # Devuelve un array con forma (num_coils, 3, 4*num_seg)

    
    def polygonal_spires(self, num_seg, n=5):
        """
        Generates coordinates for multiple polygonal coils in 3D.

        Parameters:
            num_seg (int): Total number of segments.
            n (int): Number of sides of the polygon (e.g., 5 = pentagon).

        Returns:
            numpy.ndarray: Array of shape (num_coils, 3, total_num_seg).
        """
        total_num_seg = n * num_seg  # Ensure all sides have an equal number of points
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)  # Polygon vertex angles

        spires = []

        for coil_idx in range(self.coils_number):
            r = self.L[coil_idx] / 2
            y_vertices = r * np.sin(angles)
            z_vertices = r * np.cos(angles)

            # Close the polygon by repeating the first vertex at the end
            y_closed = np.append(y_vertices, y_vertices[0])
            z_closed = np.append(z_vertices, z_vertices[0])

            # Generate all edges continuously
            all_edges = []
            for j in range(n):
                y_edge = np.linspace(y_closed[j], y_closed[j+1], num_seg)
                z_edge = np.linspace(z_closed[j], z_closed[j+1], num_seg)
                x_edge = np.zeros(num_seg)
                all_edges.append(np.vstack((x_edge, y_edge, z_edge)))  # Shape (3, num_seg)

            spire = np.hstack(all_edges)  # Final shape (3, total_num_seg)

            displacement = np.array([self.pos[coil_idx], 0, 0]).reshape(3, 1)

            # Apply transformation
            spire = np.einsum('ij,jk->ik', np.dot(self.A, rotz_180), spire - displacement)

            spires.append(spire)

        return np.array(spires)  # Shape: (num_coils, 3, total_num_seg)


    def star_spires(self, num_seg, star_points=6):
        """
        Generates coordinates for two star-shaped coils (spires) in 3D space,
        divided into 4 equal groups of edges.
        
        Parameters:
            A (ndarray): A 3x3 transformation matrix to rotate or transform the star coordinates.
            h (float): Separation between the spires (displacement along the x-axis).
            r (float): Outer radius of the star.
            num_seg (int): Number of segments (points) to be distributed among all edges of the star.
            star_points (int): Number of star points (default is 6, yielding 12 vertices with alternating outer and inner).
            
        Returns:
            tuple: (spire1, spire2) each of shape (4, 3, group_size * seg_per_edge)
                where 4 is the number of groups,
                3 corresponds to (x, y, z),
                and group_size * seg_per_edge is the total points in each group.
        """
        total_num_seg = 4*num_seg
        total_vertices = star_points * 2  # Total vertices (e.g., 12 for a 6-point star)
        seg_per_edge = total_num_seg // total_vertices  # Ensure integer division
        
        # Generate angles for each vertex
        angles = np.linspace(0, 2*np.pi, total_vertices, endpoint=False)
        
        spires = []

        for coil_idx in range(self.coils_number): 
            r = self.L[coil_idx] / 2
            # Alternating radii for outer and inner points
            radii = np.empty(total_vertices)
            radii[0::2] = r          # Outer points (even indices)
            radii[1::2] = r / 2      # Inner points (odd indices)
        
            # Compute y and z coordinates of the vertices
            y_vertices = radii * np.sin(angles)
            z_vertices = radii * np.cos(angles)
        
            # Close the star by repeating the first vertex at the end
            y_closed = np.append(y_vertices, y_vertices[0])
            z_closed = np.append(z_vertices, z_vertices[0])
        
            # Generate edges by interpolating between consecutive vertices
            star_edges = []
            for i in range(total_vertices):
                # Linearly interpolate between the current and next vertex
                y_edge = np.linspace(y_closed[i], y_closed[i+1], seg_per_edge)
                z_edge = np.linspace(z_closed[i], z_closed[i+1], seg_per_edge)
                x_edge = np.zeros(seg_per_edge)  # X-coordinate is h/2 for the first spire
            
                # Stack coordinates into (seg_per_edge, 3) array and add to the list
                edge_coords = np.vstack((x_edge, y_edge, z_edge)).T
                star_edges.append(edge_coords)
        
            # Convert list of edges to a numpy array (shape: [3, total_vertices*seg_per_edge])
            star_edges = np.array(star_edges)
            sides = star_edges.reshape(-1, 3).T
        
            # Compute coordinates for the second spire (shifted by -h along x-axis before transformation)
            displacement = np.array([self.pos[coil_idx], 0, 0]).reshape(3, 1)
            spire = np.einsum('ij,jk->ik', np.dot(self.A, rotz_180), sides - displacement)
            spires.append(spire)
        spires = np.array(spires)
        return spires


    def __repr__(self):
        return (f"CoilParameters(coils_number={self.coils_number}, L={self.L}, h={self.h}, "
                f"N={self.N}, I={self.I}, A_shape={self.A.shape})")


def generate_range(x_range, y_range=None, z_range=None, step_size_x=0.1, step_size_y=None, step_size_z=None):
    """
    Generates a sorted NumPy array of values covering the given ranges with a specified step size.
    If only x_range is provided, y_range and z_range will be set to x_range.
    If only step_size_x is provided, step_size_y and step_size_z will be set to [-0, 0].

    Parameters:
        x_range, y_range, z_range (tuple): (min, max) values for each axis.
        step_size_x, step_size_y, step_size_z (float): Step sizes for each axis.

    Returns:
        X_unique, Y_unique, Z_unique (numpy.ndarray): Unique coordinates in the XY, YZ, and XZ planes.
    """
    # If y_range and z_range are not provided, set them equal to x_range
    if y_range is None:
        y_range = [-0, 0]
    if z_range is None:
        z_range = [-0, 0]

    # If step_size_y and step_size_z are not provided, set them equal to step_size_x
    if step_size_y is None:
        step_size_y = step_size_x
    if step_size_z is None:
        step_size_z = step_size_x

    # Generate values from -2*a to 2*a with a step size of step_size
    range_vals_x = np.arange(x_range[0], x_range[1] + step_size_x, step_size_x)
    range_vals_y = np.arange(y_range[0], y_range[1] + step_size_y, step_size_y)
    range_vals_z = np.arange(z_range[0], z_range[1] + step_size_z, step_size_z)
    
    # Ensure critical points (-2*a, 0, and 2*a) are included in the range
    critical_vals_x = np.array([x_range[0], 0, x_range[1]])
    critical_vals_y = np.array([y_range[0], 0, y_range[1]])
    critical_vals_z = np.array([z_range[0], 0, z_range[1]])
    range_vals_x = np.unique(np.concatenate((range_vals_x, critical_vals_x)))
    range_vals_y = np.unique(np.concatenate((range_vals_y, critical_vals_y)))
    range_vals_z = np.unique(np.concatenate((range_vals_z, critical_vals_z)))
    
    # Create a meshgrid for the XY, YZ, and XZ planes
    X, Y = np.meshgrid(range_vals_x, range_vals_y)
    
    # Points in the XY plane (Z = 0)
    X_xy, Y_xy = X.flatten(), Y.flatten()
    Z_xy = np.zeros_like(X_xy)
    
    # Points in the YZ plane (X = 0)
    Y, Z = np.meshgrid(range_vals_y, range_vals_z)    
    Y_yz, Z_yz = Y.flatten(), Z.flatten()
    X_yz = np.zeros_like(Y_yz)
    
    # Points in the XZ plane (Y = 0)
    X, Z = np.meshgrid(range_vals_x, range_vals_z)    
    X_xz, Z_xz = X.flatten(), Z.flatten()
    Y_xz = np.zeros_like(X_xz)
    
    # Concatenate all points
    X_total = np.concatenate([X_xy, X_yz, X_xz])
    Y_total = np.concatenate([Y_xy, Y_yz, Y_xz])
    Z_total = np.concatenate([Z_xy, Z_yz, Z_xz])
    
    # Combine coordinates into a single array of shape (N, 3)
    points = np.column_stack((X_total, Y_total, Z_total))
    
    # Remove duplicate points
    unique_points = np.unique(points, axis=0)
    
    # Split the unique points back into X, Y, Z
    X_unique, Y_unique, Z_unique = unique_points[:, 0], unique_points[:, 1], unique_points[:, 2]
    
    return X_unique, Y_unique, Z_unique


def calculate_field(args):
    """
    Calculates the magnetic field at a given point due to a current-carrying coil.
    
    Parameters:
        args (tuple): A tuple containing:
            A1 (float): Proportionality constant for the magnetic field (e.g., permeability times current).
            P (numpy.ndarray): The point in 3D space where the magnetic field is calculated (shape: (3,)).
            side (numpy.ndarray): 3D coordinates of the coil segments (shape: (num_sides, 3, num_points)).
    
    Returns:
        numpy.ndarray: The magnetic field vector (shape: (3,)).
    """
    A1, P, side = args
    B = np.zeros(3)  # Initialize the magnetic field vector to zero
    dl = np.diff(side, axis=1)  # Differential length elements for each segment of the coil
    
 # Try to compute the differential length elements for each segment of the coil
    try:
        # Loop over each differential length element in the current side
        for j in range(dl.shape[1]):
            # Vector from the differential element to the point of interest
            R = P - side[:, j]
            d1 = dl[:, j]
            # Calculate the contribution to the magnetic field (Biot-Savart Law)
            dB = (A1[j]) * np.cross(d1, R) / np.linalg.norm(R)**3
                
            # Accumulate the contribution to the total magnetic field
            B += dB
    except Exception as e:
        print("Error: ", e)
        raise
    return B


def magnetic_field_coil_parallel(P, N_arr, I, coils, n):
    """
    Calculates the magnetic field at observation points P due to two square coils
    using the Biot-Savart Law.

    Parameters:
        P (np.ndarray): Observation points where the magnetic field is calculated (matrix of size m x 3).
        N (int): Number of turns in each coil.
        I (float): Current flowing through each coil.
        coils (np.ndarray): 3D coordinates of the coil segments (array of size num_seg x 3 x num_points).
        n (int): Number of segments per coil to process simultaneously.

    Returns:
        np.ndarray: Total magnetic field (B) calculated at each observation point P (matrix of size m x 3).
    """
    # Proportionality constant from the Biot-Savart Law
    A1 = (N_arr * MU_0 * I) / (4 * np.pi)

    # Use multiprocessing to calculate in parallel
    with Pool(processes=cpu_count()) as pool:
        B_segments = []  # List to store magnetic field results for each observation point
        
        # Iterate over each observation point in P
        for i in range(P.shape[0]):  # P has m rows, each representing a point in space
        
            args = [(A1[j:j+n], P[i, :], coils[:, j:j+n]) for j in range(0, coils.shape[1], n)]

            # Compute the magnetic field in parallel for coil segments
            B_segments.append(pool.map(calculate_field, args))

        # Sum contributions from all segments to get the total field at each point P
        B_results = [np.sum(segments, axis=0) for segments in B_segments]

    # Return results as a NumPy array
    return np.array(B_results)

def coil_simulation_parallel(X, Y, Z, coil_params, spires_np, batch_size, enable_progress_bar=True, n=100):
    """
        Simulates the magnetic field generated by two coils on a 1D grid in three orthogonal planes.

        Parameters:
            X, Y, Z (np.ndarray): Arrays representing the coordinates of the grid points.
            coil_params (object): Parameters of the coils, including number of turns (N), and other coil properties.
            spires_np (np.ndarray): 3D coordinates of the coils (shape: num_segments x 3 x num_points).
            batch_size (int): Number of points to process in each batch.
            enable_progress_bar (bool): Whether to display a progress bar during simulation.
            n (int): Additional parameter for the simulation (may relate to precision or integration steps).

        Returns:
            pd.DataFrame: A DataFrame containing the grid coordinates and magnetic field components.
                        Columns: ['X', 'Y', 'Z', 'Bx', 'By', 'Bz'].
        """
    
    # map the spides differential with number of turns
    N_arr = np.repeat(coil_params.N, np.repeat(spires_np.shape[2], spires_np.shape[0]))

    # Generate the X-Y grid based on the coil dimensions and grid step size    
    result = []  # List to store the simulation results

    #coils = np.concatenate(spires_np, axis=0)  # Merge along the first axis
    coils = spires_np.transpose(1, 0, 2).reshape(3, -1)
    
    # Flatten X and Y arrays for easier iteration over the grid points
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()

    # Total number of grid points
    num_points = len(X_flat)
    
    # Calculate the total number of iterations for progress tracking
    num_iter = (num_points // batch_size) + (1 if num_points % batch_size != 0 else 0)

    # Initialize the progress bar
    progress_bar = tqdm(total=num_iter, desc="Simulation Progress", disable=not enable_progress_bar)

    # Iterate over the grid points in batches
    for k in range(0, len(X_flat), batch_size):
        X_batch = X_flat[k: k + batch_size]  # Batch of X coordinates
        Y_batch = Y_flat[k: k + batch_size]  # Batch of Y coordinates
        Z_batch = Z_flat[k: k + batch_size]  # Batch of Y coordinates

        # Generate 3D points for each batch in three orthogonal planes
        P_batch = np.stack([X_batch, Y_batch, Z_batch], axis=1)

        # Calculate the magnetic field at the batch points
        B = magnetic_field_coil_parallel(P_batch, N_arr, coil_params.I, coils, n)

        # Store the results in the format (X, Y, Z, Bx, By, Bz)
        result += list(zip(P_batch[:, 0], P_batch[:, 1], P_batch[:, 2], B[:, 0], B[:, 1], B[:, 2]))

        # Update the progress bar
        progress_bar.update(1)

    if progress_bar.n < progress_bar.total:
        progress_bar.update(progress_bar.total - progress_bar.n)

    # Close the progress bar once the simulation is complete
    progress_bar.close()

    # Convert the results to a DataFrame for easier data manipulation and visualization
    return pd.DataFrame(result, columns=['X', 'Y', 'Z', 'Bx', 'By', 'Bz'])

def coil_X_symmetric_simulation(X, Y, Z, coil_params, spires_np, batch_size, enable_progress_bar=True, n=100):
    """
    Simulates a coil field while assuming symmetry over the X-axis.

    Parameters:
    - X, Y, Z: Arrays representing the spatial coordinates.
    - coil_params: Parameters of the coil.
    - spires_np: Number of spires.
    - batch_size: Number of elements processed in parallel.
    - enable_progress_bar: Boolean to show progress.
    - n: Number of iterations.

    Returns:
    - A DataFrame with the original and symmetrically extended data.
    """

    # Find all indices where X is zero
    idx_zeros = np.where(X == 0)[0][0]

    # Ensure there are zeros to maintain symmetry
    if idx_zeros.size == 0:
        raise ValueError("No zeros found in X. Symmetry cannot be applied.")

    # Last zero index to include all relevant values
    idx_fin = np.where(X == 0)[0][-1]

    # Cut X, Y, Z up to include all zeros
    X_cropped = X[:idx_fin+1] if idx_zeros.size > 0 else X
    Y_cropped = Y[:idx_fin+1] if idx_zeros.size > 0 else Y
    Z_cropped = Z[:idx_fin+1] if idx_zeros.size > 0 else Y

    #f0 = None
    #hplot.plot_grid(X_cropped, Y_cropped, Z_cropped, f0)

    # Run the simulation with the reduced dataset
    result = coil_simulation_parallel(X_cropped, Y_cropped, Z_cropped, coil_params, spires_np, batch_size, enable_progress_bar, n)

    # Ensure result is a DataFrame
    if not isinstance(result, pd.DataFrame):
        raise TypeError("Expected result to be a Pandas DataFrame")

    # Reflection in the XZ plane (invert Y )
    plane_XZ = result.copy()
    plane_XZ["X"] *= -1
    plane_XZ["Bx"] *= 1

    # Reflection in the XY plane (invert Z )
    plane_XY = result.copy()
    plane_XY["X"] *= -1
    plane_XY["Bx"] *= 1

    # Combine all results, remove duplicates, and sort by coordinates
    result_final = pd.concat([result, plane_XZ, plane_XY]).drop_duplicates()

    return result_final


#def coil_symmetric_simulation(X, Y, Z, coil_params, spires_np, batch_size, enable_progress_bar=True, n=100):
    """
    #Simula el campo de una bobina asumiendo simetría sobre el eje X.
    
    #Parámetros:
    #- X, Y, Z: Arrays con las coordenadas espaciales.
    #- coil_params: Parámetros de la bobina.
    #- spires_np: Número de espiras.
    #- batch_size: Número de elementos procesados en paralelo.
    #- enable_progress_bar: Muestra la barra de progreso.
    #- n: Número de iteraciones.
    
    #Retorna:
    #- DataFrame con los datos originales y extendidos simétricamente.
    """
 #   # Máscaras para seleccionar solo los puntos en los planos reducidos
 #   mask_yz = (X == 0) & (Y >= 0)  # Mitad positiva del plano YZ
 #   mask_other_planes = (Z == 0) & (X >= 0) | (Y == 0) & (X >= 0)  # XY y XZ
    
    # Aplicar máscaras
 #   X_YZ, Y_YZ, Z_YZ = X[mask_yz], Y[mask_yz], Z[mask_yz]
 #   X_other, Y_other, Z_other = X[mask_other_planes], Y[mask_other_planes], Z[mask_other_planes]
    
    # Ejecutar simulación para YZ
 #  result_YZ = coil_simulation_parallel(X_YZ, Y_YZ, Z_YZ, coil_params, spires_np, batch_size, enable_progress_bar, n)
    
    # Reflejo del campo en el plano XZ (invertir Y)
 #   plane_YZ = result_YZ.copy()
 #   plane_YZ["Y"] *= -1
    
    # Unir los resultados de YZ
 #   result_YZ_complete = pd.concat([result_YZ, plane_YZ], ignore_index=True).drop_duplicates()
    
    # Simulación en los otros planos
 #   result_other = coil_simulation_parallel(X_other, Y_other, Z_other, coil_params, spires_np, batch_size, enable_progress_bar, n)
    
    # Aplicar simetrías en XY y XZ
 #   plane_XZ = result_other.copy()
 #   plane_XZ["X"] *= -1
    
 #   plane_XY = result_other.copy()
 #   plane_XY["X"] *= -1
    
    # Unir todos los resultados
 #   result_final = pd.concat([result_other, plane_XZ, plane_XY], ignore_index=True).drop_duplicates()
 #   result = pd.concat([result_YZ_complete, result_final], ignore_index=True).drop_duplicates()
    
 #   return result
