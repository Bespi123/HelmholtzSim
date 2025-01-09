import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

# Define global constants
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space

# Coil parameters definition
class CoilParameters:
    def __init__(self, length, height, turns):
        self.L = np.array(length)   # Helmholtz testbed-length side
        self.a = self.L / 2         # Half Helmholtz testbed length side
        self.h = np.array(height)   # Separation among Helmholtz coils
        self.N = np.array(turns)    # Number of turns in Helmholtz coils

# Program functions
def generate_range(a, grid_number):
    """
    Generates a sorted NumPy array of values ranging from -2*a to 2*a with a specified step size.
    Includes critical points -2*a, 0, and 2*a.
    """
    range_vals = np.arange(-2 * a, 2 * a, grid_number)
    critical_vals = np.array([-2 * a, 0, 2 * a])
    range_vals = np.unique(np.concatenate((range_vals, critical_vals)))
    # Define the coil geometry and calculate fields
    X, Y = np.meshgrid(range_vals, range_vals)

    return X, Y

def square_spires(A, h, a, num_seg):
    """
    Generates coordinates of two square coils in 3D space.
    """
    h_half = h / 2
    L_half = a
    y_coords = np.linspace(L_half, -L_half, num_seg)
    z_coords = np.linspace(-L_half, L_half, num_seg)

    sides = np.array([
        [h_half * np.ones(num_seg), y_coords, L_half * np.ones(num_seg)],  # Side 1
        [h_half * np.ones(num_seg), -L_half * np.ones(num_seg), y_coords], # Side 2
        [h_half * np.ones(num_seg), z_coords, -L_half * np.ones(num_seg)], # Side 3
        [h_half * np.ones(num_seg), L_half * np.ones(num_seg), z_coords]   # Side 4
    ])

    spire1 = np.einsum('ij,ljk->lik', A, sides)
    displacement = np.array([h, 0, 0]).reshape(3, 1)
    spire2 = np.einsum('ij,ljk->lik', A, sides - displacement[None, :, :])

    return spire1, spire2

def circular_spires(A, h, r, num_seg):
    """
    Generates coordinates of two circular coils (spirals) in 3D space, divided into four quadrants.
    
    Parameters:
    A (ndarray): A 3x3 transformation matrix.
    h (float): The height (length along the z-axis) of the spirals.
    r (float): The radius of the circular spirals.
    num_seg (int): The number of segments (points) along the spirals.
    
    Returns:
    tuple: Two arrays containing the coordinates of the two spirals, divided into four quadrants.
    """
    # Generate the full spiral
    x_vals = h/2 * np.ones(4*num_seg)  # height values for the spiral
    theta_vals = np.linspace(0, 2 * np.pi, 4*num_seg)  # angle values for the circular motion
    
    # Parametrize the circular spiral in 3D
    x_coords = x_vals  # x = r * cos(theta)
    y_coords = r * np.sin(theta_vals)  # y = r * sin(theta)
    z_coords = r * np.cos(theta_vals)  # z = linear progression along the z-axis
    
    # Stack the coordinates to form the spiral shape
    spiral_coords = np.array([x_coords, y_coords, z_coords]).T  # Shape: (num_seg, 3)

    # Divide the spiral into four quadrants (or "sides")
    # Each side corresponds to one quadrant of the spiral
    half_num_seg = num_seg

    sides = np.array([
        spiral_coords[:half_num_seg],  # First quadrant
        spiral_coords[half_num_seg:2*half_num_seg],  # Second quadrant
        spiral_coords[2*half_num_seg:3*half_num_seg],  # Third quadrant
        spiral_coords[3*half_num_seg:],  # Fourth quadrant
    ])
    
    sides = sides.transpose(0, 2, 1)
    
    # Apply transformation to the first spiral (spire1)
    spire1 = np.einsum('ij,ljk->lik', A, sides)
    displacement = np.array([h, 0, 0]).reshape(3, 1)
    spire2 = np.einsum('ij,ljk->lik', A, sides - displacement[None, :, :])

    # Return both spirals
    return spire1, spire2

def calculate_field(args):
    #A1, P, side = args
    #B = np.zeros(3)
    #dl = np.diff(side)  # Elemento de longitud diferencial para el segmento

    #for j in range(dl.shape[1]):
    #    R = P - side[:,j]
    #    dB = (A1) * np.cross(dl[:, j], R) / np.linalg.norm(R)**3
    #    B += dB
    #return B
    A1, P, side = args
    B = np.zeros(3)
    dl = np.diff(side, axis=2)  # Elemento de longitud diferencial para el segmento
    
    for k in range(side.shape[0]):
      for j in range(dl.shape[2]):
          R = P - side[k,:,j]
          dB = (A1) * np.cross(dl[k,:, j], R) / np.linalg.norm(R)**3
          B += dB
    return B

def magnetic_field_square_coil_parallel(P, N, I, coils, n):
    """
    Calcula el campo magnético en el punto P debido a dos bobinas cuadradas utilizando la Ley de Biot-Savart.

    Parámetros:
        P (np.ndarray): Punto de observación donde se calcula el campo magnético (vector 3x1).
        N (int): Número de vueltas en cada bobina.
        I (float): Corriente que fluye a través de cada bobina.
        spire1, spire2 (dict): Diccionarios que representan cada bobina, con campos para las coordenadas 3D de los segmentos.

    Retorna:
        tuple: Campo magnético total (B), campo de spire1 (B1) y campo de spire2 (B2) como arreglos numpy.
    """
    A1 = (N * MU_0 * I )/ (4 * np.pi)

    with Pool(processes=cpu_count()) as pool:
        # Preparar argumentos para cada columna de P y calcular en paralelo
        B_segments = []
        
        for i in range(P.shape[0]):  # Itera sobre las columnas de P (0, 1, 2)
            args = [(A1, P[i, :], coils[j:j+n, :, :]) for j in range(0, coils.shape[0], n)]
            B_segments.append(pool.map(calculate_field, args))

        # Sumar las contribuciones de todos los segmentos para cada campo
        B_results = [np.sum(segments, axis=0) for segments in B_segments]

    # Retornar los resultados desglosados
    return B_results

def coil_simulation_1d_sequential(grid_number, coil_params, current, coil1, coil2, parallel_coils, batch_size):
    X, Y = generate_range(coil_params.a, grid_number)
    # Define the coil geometry and calculate fields
    #X, Y = np.meshgrid(range_vals, range_vals)
    # Initialize coil dictionary to store results
    coil = {}
    # Define storage for results
    coil['xy'] = {'X': X, 'Y': Y, 'Bx': np.nan * np.ones_like(X), 'By': np.nan * np.ones_like(Y), 'Bz': np.nan * np.ones_like(Y), 'norB': np.nan * np.ones_like(Y)}
    coil['yz'] = {'Y': X, 'Z': Y, 'Bx': np.nan * np.ones_like(X), 'By': np.nan * np.ones_like(Y), 'Bz': np.nan * np.ones_like(Y), 'norB': np.nan * np.ones_like(Y)}
    coil['xz'] = {'X': X, 'Z': Y, 'Bx': np.nan * np.ones_like(X), 'By': np.nan * np.ones_like(Y), 'Bz': np.nan * np.ones_like(Y), 'norB': np.nan * np.ones_like(Y)}
    
    #coil1, coil2 = square_spires(A, coil_params.h, coil_params.a, num_seg)
    coils = np.concatenate([coil1, coil2], axis=0)

    # Loop through the grid to calculate the magnetic field
    num_iter = len(X) ** 2
    current_iter = 0

    # Initialize a progress bar
    progress_bar = tqdm(total=num_iter, desc="Simulation Progress")

    points = []

    
    # Convertir X e Y en arreglos 1D para facilitar la iteración
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    m = X.shape[1]  # Número de columnas de X y Y (3 en este caso)
    
    # Agrupar `n` elementos de `X` y `Y` en cada iteración
    for k in range(0, len(X_flat), batch_size):
        X_batch = X_flat[k: k + batch_size]
        Y_batch = Y_flat[k: k + batch_size]

        i = k // m  # Calcular el índice de fila i
        j = k % m   # Calcular el índice de columna j

        # Generar los puntos para los lotes
        P_batch = np.stack([
            np.stack([X_batch, Y_batch, np.zeros_like(X_batch)], axis=1),  # P1 X-Y plane
            np.stack([np.zeros_like(X_batch), X_batch, Y_batch], axis=1),  # P2 Y-Z plane
            np.stack([X_batch, np.zeros_like(X_batch), Y_batch], axis=1)   # P3 X-Z plane
        ], axis=1).reshape(-1, 3)    # Forma final: (batch_size, 3, 3)
        
        B = magnetic_field_square_coil_parallel(P_batch, coil_params.N, current, coils, parallel_coils)

        for l in range(0, batch_size):
            # Evaluate magnetic field in the X-Y plane
            if i < coil['xy']['Bx'].shape[0] and j + l < coil['xy']['Bx'].shape[1]:
                coil['xy']['Bx'][i, j + l], coil['xy']['By'][i, j + l], coil['xy']['Bz'][i, j + l]= B[l*3]
                coil['xy']['norB'][i, j + l] = np.linalg.norm(B[(l*(3))])

            # Evaluate magnetic field in the Y-Z plane
            if i < coil['xy']['Bx'].shape[0] and j + l < coil['xy']['Bx'].shape[1]:
                coil['yz']['Bx'][i, j + l], coil['yz']['By'][i, j + l], coil['yz']['Bz'][i, j + l] = B[(l*(3)+1)]
                coil['yz']['norB'][i, j +l] = np.linalg.norm(B[(l*(3)+1)])

            # Evaluate magnetic field in the X-Z plane
            if i < coil['xy']['Bx'].shape[0] and j + l < coil['xy']['Bx'].shape[1]:
                coil['xz']['Bx'][i, j + l], coil['xz']['By'][i, j + l], coil['xz']['Bz'][i, j + l] = B[(l*(3)+2)]
                coil['xz']['norB'][i, j + l] = np.linalg.norm(B[(l*(3)+2)])

        # Update progress bar
        current_iter += 1
        progress_bar.update(batch_size)

    # Close the progress bar once the simulation is complete
    progress_bar.close()

    return coil