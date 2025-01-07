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
    return np.unique(np.concatenate((range_vals, critical_vals)))

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

def calculate_segment_field(args):
    A1, P, side = args
    B = np.zeros(3)
    dl = np.diff(side)  # Elemento de longitud diferencial para el segmento
    #Pv = np.tile(P, (side.shape[1], 1)).T
    #R  = Pv - side

    for j in range(dl.shape[1]):
        R = P - side[:,j]
        #dB = (A1) * np.cross(dl[:, j], R[:,j]) / np.linalg.norm(R)**3
        dB = (A1) * np.cross(dl[:, j], R) / np.linalg.norm(R)**3
        B += dB

    return B

def magnetic_field_square_coil_parallel(P, N, I, spire1, spire2):
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
        # Preparar argumentos para cada segmento de spire1 y spire2
        args1 = [(A1, P, side) for side in spire1]
        args2 = [(A1, P, side) for side in spire2]

        # Calcular campos magnéticos en paralelo
        B1_segments = pool.map(calculate_segment_field, args1)
        B2_segments = pool.map(calculate_segment_field, args2)

    # Sumar las contribuciones de todos los segmentos
    B1 = np.sum(B1_segments, axis=0)
    B2 = np.sum(B2_segments, axis=0)
    B = B1 + B2

    return B, B1, B2

def coil_simulation_1d_sequential(grid_number, A, coil_params, current, num_seg):
    range_vals = generate_range(coil_params.a, grid_number)
    # Define the coil geometry and calculate fields
    X, Y = np.meshgrid(range_vals, range_vals)
    # Initialize coil dictionary to store results
    coil = {}
    # Define storage for results
    coil['xy'] = {'X': X, 'Y': Y, 'Bx': np.nan * np.ones_like(X), 'By': np.nan * np.ones_like(Y), 'Bz': np.nan * np.ones_like(Y), 'norB': np.nan * np.ones_like(Y)}
    coil['yz'] = {'Y': X, 'Z': Y, 'Bx': np.nan * np.ones_like(X), 'By': np.nan * np.ones_like(Y), 'Bz': np.nan * np.ones_like(Y), 'norB': np.nan * np.ones_like(Y)}
    coil['xz'] = {'X': X, 'Z': Y, 'Bx': np.nan * np.ones_like(X), 'By': np.nan * np.ones_like(Y), 'Bz': np.nan * np.ones_like(Y), 'norB': np.nan * np.ones_like(Y)}
    
    coil['spire1'], coil['spire2'] = square_spires(A, coil_params.h, coil_params.a, num_seg)

    # Loop through the grid to calculate the magnetic field
    num_iter = len(X) ** 2
    current_iter = 0

    # Initialize a progress bar
    progress_bar = tqdm(total=num_iter, desc="Simulation Progress")

    for i in range(len(X)):  # Loop over X values
        for j in range(len(Y)):  # Loop over Y values
            # Evaluate magnetic field in the X-Y plane
            P1 = np.array([X[i, j], Y[i, j], 0])
            B1, _, _ = magnetic_field_square_coil_parallel(P1, coil_params.N, current, coil['spire1'], coil['spire2'])
            coil['xy']['Bx'][i, j] = B1[0]
            coil['xy']['By'][i, j] = B1[1]
            coil['xy']['Bz'][i, j] = B1[2]
            coil['xy']['norB'][i, j] = np.linalg.norm(B1)

            # Evaluate magnetic field in the Y-Z plane
            P2 = np.array([0, X[i, j], Y[i, j]])
            B2, _, _ = magnetic_field_square_coil_parallel(P2, coil_params.N, current, coil['spire1'], coil['spire2'])
            coil['yz']['Bx'][i, j] = B2[0]
            coil['yz']['By'][i, j] = B2[1]
            coil['yz']['Bz'][i, j] = B2[2]
            coil['yz']['norB'][i, j] = np.linalg.norm(B2)

            # Evaluate magnetic field in the X-Z plane
            P3 = np.array([X[i, j], 0, Y[i, j]])
            B3, _, _ = magnetic_field_square_coil_parallel(P3, coil_params.N, current, coil['spire1'], coil['spire2'])
            coil['xz']['Bx'][i, j] = B3[0]
            coil['xz']['By'][i, j] = B3[1]
            coil['xz']['Bz'][i, j] = B3[2]
            coil['xz']['norB'][i, j] = np.linalg.norm(B3)

            # Update progress bar
            current_iter += 1
            progress_bar.update(1)

    # Close the progress bar once the simulation is complete
    progress_bar.close()

    return coil

# Initialize coil parameters
X_coil = CoilParameters(1.03, 0.85, 36)
I = 1
grid_length_size = 0.05
num_seg = 1000
Ax = np.eye(3)

# Generate data and process
start_time = time.time()
x_coil_results = coil_simulation_1d_sequential(grid_length_size, Ax, X_coil, I, num_seg)
end_time = time.time()

# Calcular y mostrar el tiempo de ejecución
execution_time = end_time - start_time
print(f"Tiempo de ejecución p: {execution_time} segundos")

# Save the x_coil_results dictionary using numpy.save
np.save('x_coil_resultspara.npy', x_coil_results)
# Display results
print("First results:", x_coil_results[:5])