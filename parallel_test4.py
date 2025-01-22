# Import dependencies
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from scipy.optimize import minimize  # Optimization function from SciPy
#import parallel_test4 as simulation  # Custom simulation module
#import numpy as np  # Numerical computations

# Define global constants
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space

# Coil parameters definition
class CoilParameters:
    def __init__(self, length, height, turns):
        # Initialize the length of the Helmholtz testbed side
        self.L = np.array(length)   # Helmholtz testbed-length side
        # Calculate and store half the Helmholtz testbed length side
        self.a = self.L / 2         # Half Helmholtz testbed length side
        # Initialize the separation between Helmholtz coils
        self.h = np.array(height)   # Separation among Helmholtz coils
        # Initialize the number of turns in the Helmholtz coils
        self.N = np.array(turns)    # Number of turns in Helmholtz coils


# Program functions
def generate_range(a, grid_number):
    """
    Generates a sorted NumPy array of values ranging from -2*a to 2*a with a specified step size.
    Includes critical points -2*a, 0, and 2*a.
    
    Parameters:
        a (float): Half the Helmholtz testbed length side.
        grid_number (float): Step size for generating the range of values.
    
    Returns:
        X, Y (numpy.ndarray): Meshgrid arrays representing the x and y coordinates for the range.
    """
    # Generate values from -2*a to 2*a with a step size of grid_number
    range_vals = np.arange(-2 * a, 2 * a, grid_number)
    
    # Ensure critical points (-2*a, 0, and 2*a) are included in the range
    critical_vals = np.array([-2 * a, 0, 2 * a])
    
    # Combine the range values and critical points, ensuring uniqueness and sorting
    range_vals = np.unique(np.concatenate((range_vals, critical_vals)))
    
    # Create a meshgrid for defining the coil geometry and subsequent field calculations
    X, Y = np.meshgrid(range_vals, range_vals)
    
    return X, Y


def square_spires(A, h, a, num_seg):
    """
    Generates coordinates of two square coils in 3D space.

    Parameters:
        A (numpy.ndarray): Transformation matrix to rotate or transform the coil's coordinates.
        h (float): Separation between the two coils.
        a (float): Half the length of the square coil side.
        num_seg (int): Number of segments per side of the square.

    Returns:
        spire1, spire2 (numpy.ndarray): Arrays containing the 3D coordinates of the two square coils.
    """
    # Calculate half the separation distance between the two coils
    h_half = h / 2
    
    # Half the length of the square side
    L_half = a
    
    # Generate evenly spaced points for the y and z coordinates along the square sides
    y_coords = np.linspace(L_half, -L_half, num_seg)
    z_coords = np.linspace(-L_half, L_half, num_seg)
    
    # Define the 3D coordinates of the four sides of a square coil
    sides = np.array([
        [h_half * np.ones(num_seg), y_coords, L_half * np.ones(num_seg)],  # Side 1: top edge
        [h_half * np.ones(num_seg), -L_half * np.ones(num_seg), y_coords], # Side 2: right edge
        [h_half * np.ones(num_seg), z_coords, -L_half * np.ones(num_seg)], # Side 3: bottom edge
        [h_half * np.ones(num_seg), L_half * np.ones(num_seg), z_coords]   # Side 4: left edge
    ])
    
    # Transform the coordinates of the first coil using the matrix A
    spire1 = np.einsum('ij,ljk->lik', A, sides)
    
    # Create a displacement vector to position the second coil
    displacement = np.array([h, 0, 0]).reshape(3, 1)
    
    # Transform the coordinates of the second coil using the matrix A and apply the displacement
    spire2 = np.einsum('ij,ljk->lik', A, sides - displacement[None, :, :])
    
    # Return the 3D coordinates of both square coils
    return spire1, spire2


def circular_spires(A, h, r, num_seg):
    """
    Generates coordinates of two circular coils (spirals) in 3D space, divided into four quadrants.
    
    Parameters:
        A (ndarray): A 3x3 transformation matrix to rotate or transform the spiral coordinates.
        h (float): The height (length along the z-axis) of the spirals.
        r (float): The radius of the circular spirals.
        num_seg (int): The number of segments (points) for each quarter of the spiral.
    
    Returns:
        tuple: Two arrays containing the coordinates of the two spirals, divided into four quadrants.
    """
    # Generate height values for the spiral along the x-axis
    x_vals = h / 2 * np.ones(4 * num_seg)  # Constant height for all points
    
    # Generate angle values for the circular motion (theta)
    theta_vals = 2 * np.pi - np.linspace(0, 2 * np.pi, 4 * num_seg)  # Full circle (360°)
    
    # Parametrize the circular spiral in 3D space
    x_coords = x_vals                    # x remains constant
    y_coords = r * np.sin(theta_vals)    # y = r * sin(theta), circular pattern in the y-direction
    z_coords = r * np.cos(theta_vals)    # z = r * cos(theta), circular pattern in the z-direction
    
    # Stack the coordinates into a 2D array (shape: [4*num_seg, 3])
    spiral_coords = np.array([x_coords, y_coords, z_coords]).T
    
    # Divide the spiral into four quadrants, each containing `num_seg` points
    half_num_seg = num_seg
    sides = np.array([
        spiral_coords[:half_num_seg],           # First quadrant
        spiral_coords[half_num_seg:2*half_num_seg],  # Second quadrant
        spiral_coords[2*half_num_seg:3*half_num_seg],  # Third quadrant
        spiral_coords[3*half_num_seg:],        # Fourth quadrant
    ])
    
    # Transpose the sides to match the required shape for transformation (shape: [4, 3, num_seg])
    sides = sides.transpose(0, 2, 1)
    
    # Apply the transformation matrix A to the coordinates of the first spiral (spire1)
    spire1 = np.einsum('ij,ljk->lik', A, sides)
    
    # Define the displacement vector to position the second spiral
    displacement = np.array([h, 0, 0]).reshape(3, 1)
    
    # Apply the transformation and displacement to define the second spiral (spire2)
    spire2 = np.einsum('ij,ljk->lik', A, sides - displacement[None, :, :])
    
    # Return the coordinates of both spirals
    return spire1, spire2


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
    dl = np.diff(side, axis=2)  # Differential length elements for each segment of the coil
    
    # Loop over each side of the coil
    for k in range(side.shape[0]):
        # Loop over each differential length element in the current side
        for j in range(dl.shape[2]):
            # Vector from the differential element to the point of interest
            R = P - side[k, :, j]
            
            # Calculate the contribution to the magnetic field (Biot-Savart Law)
            dB = (A1) * np.cross(dl[k, :, j], R) / np.linalg.norm(R)**3
            
            # Accumulate the contribution to the total magnetic field
            B += dB
    
    return B


def magnetic_field_square_coil_parallel(P, N, I, coils, n):
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
    A1 = (N * MU_0 * I) / (4 * np.pi)

    # Use multiprocessing to calculate in parallel
    with Pool(processes=cpu_count()) as pool:
        B_segments = []  # List to store magnetic field results for each observation point
        
        # Iterate over each observation point in P
        for i in range(P.shape[0]):  # P has m rows, each representing a point in space
            # Prepare arguments for each segment of the coil
            args = [(A1, P[i, :], coils[j:j+n, :, :]) for j in range(0, coils.shape[0], n)]
            
            # Compute the magnetic field in parallel for coil segments
            B_segments.append(pool.map(calculate_field, args))

        # Sum contributions from all segments to get the total field at each point P
        B_results = [np.sum(segments, axis=0) for segments in B_segments]

    # Return results as a NumPy array
    return np.array(B_results)


def coil_simulation_1d_sequential(X, Y, coil_params, current, coil1, coil2, parallel_coils, batch_size):
    """
    Simulates the magnetic field generated by two coils on a 1D grid in three orthogonal planes.

    Parameters:
        grid_number (float): Step size for generating the grid.
        coil_params (CoilParameters): Parameters of the coils, including length, height, and turns.
        current (float): Current flowing through the coils.
        coil1, coil2 (np.ndarray): 3D coordinates of the two coils (arrays of shape num_segments x 3 x num_points).
        parallel_coils (int): Number of coil segments processed simultaneously.
        batch_size (int): Number of points to process in each batch.

    Returns:
        pd.DataFrame: A DataFrame containing the grid coordinates and magnetic field components.
                      Columns: ['X', 'Y', 'Z', 'Bx', 'By', 'Bz'].
    """
    # Generate the X-Y grid based on the coil dimensions and grid step size
    #X, Y = generate_range(coil_params.a, grid_number)
    
    result = []  # List to store the simulation results
    coils = np.concatenate([coil1, coil2], axis=0)  # Combine both coils into a single array

    # Calculate the total number of iterations for progress tracking
    num_iter = len(X) ** 2
    progress_bar = tqdm(total=num_iter, desc="Simulation Progress")  # Initialize a progress bar

    # Flatten X and Y arrays for easier iteration over the grid points
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    if X.ndim > 1:
        m = X.shape[1]  # Number of columns in X and Y (3 in this example)
    else:
        m = 1

    # Iterate over the grid points in batches
    for k in range(0, len(X_flat), batch_size):
        X_batch = X_flat[k: k + batch_size]  # Batch of X coordinates
        Y_batch = Y_flat[k: k + batch_size]  # Batch of Y coordinates

        # Generate 3D points for each batch in three orthogonal planes
        P_batch = np.stack([
            np.stack([X_batch, Y_batch, np.zeros_like(X_batch)], axis=1),  # P1: X-Y plane
            np.stack([np.zeros_like(X_batch), X_batch, Y_batch], axis=1),  # P2: Y-Z plane
            np.stack([X_batch, np.zeros_like(X_batch), Y_batch], axis=1)   # P3: X-Z plane
        ], axis=1).reshape(-1, 3)  # Final shape: (batch_size * 3, 3)

        # Calculate the magnetic field at the batch points
        B = magnetic_field_square_coil_parallel(P_batch, coil_params.N, current, coils, parallel_coils)

        # Store the results in the format (X, Y, Z, Bx, By, Bz)
        result += list(zip(P_batch[:, 0], P_batch[:, 1], P_batch[:, 2], B[:, 0], B[:, 1], B[:, 2]))

        # Update the progress bar
        progress_bar.update(batch_size)

    # Close the progress bar once the simulation is complete
    progress_bar.close()

    # Convert the results to a DataFrame for easier data manipulation and visualization
    return pd.DataFrame(result, columns=['X', 'Y', 'Z', 'Bx', 'By', 'Bz'])



def objective(variables, A, target_bx, grid_length_size=0.01, num_seg=100):
    """
    Objective function with additional parameters.
    
    Parameters:
    - variables: List of [length, distance] to optimize.
    - A: Transformation matrix for the spires.
    - target_bx: Target value for Bx in the optimization.
    - grid_length_size: Step size for the simulation grid (default: 0.01).
    - num_seg: Number of segments for the spires (default: 100).

    Returns:
    - Objective function value.
    """
    length, distance = variables  # Extract length and distance variables
    turns = 30  # Number of turns for the coil
    I = 1  # Current in the coil
    
    # Initialize coil parameters
    coil = CoilParameters(length, distance, turns)

    # Generate coil spires and simulation grid
    spire1_s, spire2_s = square_spires(A, coil.h, coil.a, num_seg)
    X = np.arange(-coil.a, coil.a, grid_length_size)  # X-axis points
    Y = np.zeros_like(X)  # Y-axis remains zero (1D simulation)

    # Run the simulation
    x_coil_results_s = coil_simulation_1d_sequential(
        X, Y, coil, I, spire1_s, spire2_s, 1, 20
    )
    bx_line = x_coil_results_s[
        (x_coil_results_s['Y'] == 0) & (x_coil_results_s['Z'] == 0)
    ]

    # Extract the maximum Bx value and calculate error
    bx = bx_line['Bx'].max()
    e = target_bx - bx * 1e9  # Target error calculation

    if e <= 0:
        tolerance = 0.005 * bx
        lower_bound = bx - tolerance
        upper_bound = bx + tolerance
        filtered_points = bx_line[
            (bx_line['Bx'] >= lower_bound) & (bx_line['Bx'] <= upper_bound)
        ]

        # Check for contiguity of points
        x_values = filtered_points['X'].sort_values()
        is_contiguous = all(
            (x_values.iloc[i + 1] - x_values.iloc[i]) <= 2 * grid_length_size
            for i in range(len(x_values) - 1)
        )

        # If points are not contiguous, assign an infinite penalty
        if not is_contiguous:
            return 5000  # Penalty

        # Calculate range (a) and return the negative for maximization
        a = abs(filtered_points['X'].max() - filtered_points['X'].min())
        return -a  # Negative since we minimize, but want to maximize range
    else:
        # Penalize for exceeding the target error
        return 5000 + e

# Definir los argumentos adicionales
#Ax = np.eye(3)  # Identity matrix
#target_bx = 45839.86895741893  # Desired target for Bx

# Bounds for the variables
#my_bounds = [(1e-1, 1), (0.4, 1)]  # Bounds for length and distance

# Constraints (currently empty)
#constraints = []

def optimization_fun(Ax, target_bx, my_bounds, constraints):
    # Ejecutar la optimización
    result = minimize(
        objective,
        x0=[1, 0.4],  # Initial guess
        args=(Ax, target_bx),  # Argumentos adicionales para la función objetivo
        method='SLSQP',  # Optimization method
        bounds=my_bounds,  # Variable bounds
        constraints=constraints,  # Constraints
        options={
            'maxiter': 1000,  # Maximum iterations
            'ftol': 1e-3,  # Tolerance
            'eps': 1e-1,  # Finite difference step size
            'disp': True  # Display progress
        }
    )

    # Resultados
    x, y = result.x
    print(f"Best solution: length = {x}, distance = {y}, turns = 30, Fitness = {-result.fun}")
    return x, y