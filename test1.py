import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
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

def biot_Savart_law(args):
    N = args[0]
    I = args[1]
    P = args[2:5]
    side = args[5:8]
    dl = args[8:11]

    R = P - side  # Vector de posición desde el segmento hasta el punto P
    R_norm = np.linalg.norm(R)

    if R_norm == 0:  # Evitar divisiones por cero
        return np.zeros(3)

    dB = ((N * MU_0 * I) / (4 * np.pi)) * np.cross(dl, R) / (R_norm**3)
    return dB

#def process_data_in_chunks(data, chunk_size=1000):
#    """
#    Processes data in parallel chunks to optimize memory usage.
#    """
#    results = []
#    for i in range(0, len(data), chunk_size):
#        chunk = data[i:i + chunk_size]
#        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
#            chunk_results = list(executor.map(biot_savart_law, chunk))
#        results.extend(chunk_results)
#    return np.array(results)
def process_data_in_chunks(data, chunk_size=1000):
    """
    Processes data in parallel chunks to optimize memory usage, with a progress bar.
    """
    results = []
    total_chunks = len(data) // chunk_size + (1 if len(data) % chunk_size != 0 else 0)
    
    with tqdm(total=total_chunks, desc="Processing Chunks") as pbar:
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
                chunk_results = list(executor.map(biot_Savart_law, chunk))
            results.extend(chunk_results)
            pbar.update(1)
    
    return np.array(results)


def generate_arguments(grid_number, A, coil_params, current, num_seg):
    """
    Generates arguments for Biot-Savart law computation.
    """
    range_vals = generate_range(coil_params.a, grid_number)
    X, Y = np.meshgrid(range_vals, range_vals)

    all_points = np.vstack([
        np.stack((X, Y, np.zeros_like(X)), axis=-1).reshape(-1, 3),
        np.stack((np.zeros_like(X), X, Y), axis=-1).reshape(-1, 3),
        np.stack((X, np.zeros_like(X), Y), axis=-1).reshape(-1, 3)
    ])

    coil1, coil2 = square_spires(A, coil_params.h, coil_params.L, num_seg)

    differential_1 = np.diff(coil1, axis=2)
    differential_2 = np.diff(coil2, axis=2)
    differential_1 = np.concatenate((differential_1, differential_1[:, :, -1][:, :, np.newaxis]), axis=2)
    differential_2 = np.concatenate((differential_2, differential_2[:, :, -1][:, :, np.newaxis]), axis=2)

    coils = np.vstack((coil1.swapaxes(1, 2).reshape(-1, 3), coil2.swapaxes(1, 2).reshape(-1, 3)))
    diff_coils = np.vstack((differential_1.swapaxes(1, 2).reshape(-1, 3), differential_2.swapaxes(1, 2).reshape(-1, 3)))

    all_points_repeated = np.repeat(all_points, coils.shape[0], axis=0)
    all_coils_repeated = np.tile(coils, (all_points.shape[0], 1))
    diff_coils_repeated = np.tile(diff_coils, (all_points.shape[0], 1))

    N = np.full((all_points_repeated.shape[0], 1), coil_params.N)
    I = np.full((all_points_repeated.shape[0], 1), current)

    data = np.hstack((N, I, all_points_repeated, all_coils_repeated, diff_coils_repeated))
    return data

def plot_3d_points(all_points, coils):
    """
    Plots 3D representation of points and coils.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], c='b', marker='.', s=10)
    ax.scatter(coils[:, 0], coils[:, 1], coils[:, 2], c='r', marker='.', s=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("3D Points and Coils")
    plt.show()

# Initialize coil parameters
X_coil = CoilParameters(1.03, 0.85, 36)
I = 1
grid_length_size = 0.05
num_seg = 1000
Ax = np.eye(3)

# Generate data and process
start_time = time.time()
data = generate_arguments(grid_length_size, Ax, coil_params=X_coil, current=I, num_seg=num_seg)
result = process_data_in_chunks(data)
# Marcar el tiempo de fin
end_time = time.time()
# Calcular y mostrar el tiempo de ejecución
execution_time = end_time - start_time
print(f"Tiempo de ejecución: {execution_time} segundos")

# Display results
print("First results:", result[:5])