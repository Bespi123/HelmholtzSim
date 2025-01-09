import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D
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

    if isinstance(x_coil_results, np.ndarray):
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


def simple_3d_surface_plot(coil):
    plane = 'xy'
    
    x = coil[plane]['X']
    y = coil[plane]['Y']
    norB = coil[plane]['norB']

    # Generar datos de ejemplo
    #x = np.linspace(-10, 10, 100)
    #y = np.linspace(-10, 10, 100)
    #X, Y = np.meshgrid(x, y)
    #Z = np.sqrt(X**2 + Y**2)  # Ejemplo de magnitud del campo

    # Crear gráfico de superficie 3D
    fig = go.Figure()

    fig.add_trace(go.Surface(
        z=norB,
        x=x,
        y=y,
        colorscale='Viridis',
        showscale=True
    ))

    # Configurar layout
    fig.update_layout(
        title="Gráfico de Superficie 3D",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Magnitud"
        ),
        height=700,
    )

    fig.show()

def plot_spires(spire1, spire2, color='black'):
    # Cambiar la forma de spire1 y spire2 de (4, 3, 100) a (4, 100, 3)
    spire1 = spire1.transpose(0, 2, 1)  # Cambiar la forma a (4, 100, 3)
    spire2 = spire2.transpose(0, 2, 1)  # Cambiar la forma a (4, 100, 3)
    
    # Aplana los arrays para tener una forma (num_puntos, 3) por espiral
    spire1_flat = spire1.reshape(-1, 3)  # Aplana a forma (400, 3)
    spire2_flat = spire2.reshape(-1, 3)  # Aplana a forma (400, 3)
    
    # Crear el gráfico interactivo
    trace1 = go.Scatter3d(
        x=spire1_flat[:, 0], 
        y=spire1_flat[:, 1], 
        z=spire1_flat[:, 2], 
        mode='lines', 
        line=dict(color=color, width=4),
        name='Spire 1'
    )
    
    trace2 = go.Scatter3d(
        x=spire2_flat[:, 0], 
        y=spire2_flat[:, 1], 
        z=spire2_flat[:, 2], 
        mode='lines', 
        line=dict(color=color, width=4),
        name='Spire 2'
    )
    
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        title="Helmholtz Coils"
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    
    fig.show()
    return fig

def plot_grid(X, Y, fig):
  P1_points = np.stack((X, Y, np.zeros_like(X)), axis=-1)  # X-Y plane
  P2_points = np.stack((np.zeros_like(X), X, Y), axis=-1)  # Y-Z plane
  P3_points = np.stack((X, np.zeros_like(X), Y), axis=-1)  # X-Z plane

  # Create a 3D scatter plot
  #fig = go.Figure()

  # Add points for each plane
  fig.add_trace(go.Scatter3d(
      x=P1_points[..., 0].flatten(),
      y=P1_points[..., 1].flatten(),
      z=P1_points[..., 2].flatten(),
      mode='markers',
      marker=dict(size=3, color='red'),
      name='P1 (X-Y Plane)'
  ))

  fig.add_trace(go.Scatter3d(
      x=P2_points[..., 0].flatten(),
      y=P2_points[..., 1].flatten(),
      z=P2_points[..., 2].flatten(),
      mode='markers',
      marker=dict(size=3, color='green'),
      name='P2 (Y-Z Plane)'
  ))

  fig.add_trace(go.Scatter3d(
      x=P3_points[..., 0].flatten(),
      y=P3_points[..., 1].flatten(),
      z=P3_points[..., 2].flatten(),
      mode='markers',
      marker=dict(size=3, color='blue'),
      name='P3 (X-Z Plane)'
  ))

  # Update layout for better visualization
  fig.update_layout(
      scene=dict(
          xaxis_title='X-axis',
          yaxis_title='Y-axis',
          zaxis_title='Z-axis'
      ),
      #title="Interactive 3D Plot with Plotly"
  )

  fig.show()
#file_path = '/home/bespi123/git/helmholtzCoilsDesigner/x_coil_results.npy'
#file_path = '/home/bespi123/git/x_coil_resultspara2.npy'

# Cargar el archivo
#x_coil_results = np.load(file_path, allow_pickle=True)

#plot_magnetic_field(x_coil_results)