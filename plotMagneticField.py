import numpy as np
import matplotlib.pyplot as plt

def plot_magnetic_field(x_coil_results):
    """
    Plotea los campos magnéticos generados por la simulación en los planos XY, YZ y XZ.

    Parameters:
        coil (dict): Diccionario con los resultados de los campos magnéticos en los planos.
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))

    planes = ['xy', 'yz', 'xz']
    titles = ['Plano XY', 'Plano YZ', 'Plano XZ']

    if x_coil_results.size == 1:
      coil = x_coil_results.item()  # Extraer el diccionario
    else:
      coil = x_coil_results

    for idx, plane in enumerate(planes):
        # Magnitud del campo en el plano
        ax1 = axes[idx, 0]
        norB = coil[plane]['norB']

        if plane == 'xy':
          x = coil[plane]['X']
          y = coil[plane]['Y']
        elif plane == 'yz':
          x = coil[plane]['Y']
          y = coil[plane]['Z']
        else:
          x = coil[plane]['X']
          y = coil[plane]['Z']

        im = ax1.contourf(x, y, norB, cmap='viridis', levels=100, alpha=0.7)
        fig.colorbar(im, ax=ax1)
        ax1.set_title(f"{titles[idx]}: Magnitud del campo")
        ax1.set_xlabel('X' if plane != 'yz' else 'Y')
        ax1.set_ylabel('Y' if plane == 'xy' else ('Z' if plane == 'yz' else 'X'))

        # Vectores de campo en el plano
        ax2 = axes[idx, 1]
        Bx = coil[plane]['Bx']
        By = coil[plane]['By'] if plane != 'yz' else coil[plane]['Bz']

        # Quitar nans para evitar errores
        mask = ~np.isnan(Bx) & ~np.isnan(By)
        X = x[mask]
        Y = y[mask]
        U = Bx[mask]
        V = By[mask]

        ax2.quiver(X, Y, U, V, scale=100, color='blue', alpha=0.6)
        ax2.set_title(f"{titles[idx]}: Direcciones del campo")
        ax2.set_xlabel('X' if plane != 'yz' else 'Y')
        ax2.set_ylabel('Y' if plane == 'xy' else ('Z' if plane == 'yz' else 'X'))

    plt.tight_layout()
    plt.show()

#file_path = '/home/bespi123/git/helmholtzCoilsDesigner/x_coil_results.npy'
file_path = '/home/bespi123/git/x_coil_resultspara2.npy'

# Cargar el archivo
x_coil_results = np.load(file_path, allow_pickle=True)

# Mostrar los datos
#print(x_coil_results)

plot_magnetic_field(x_coil_results)