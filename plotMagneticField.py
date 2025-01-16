import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots


def plot_magnetic_field(x_coil_results):
    """
    Plotea los campos magnéticos generados por la simulación en los planos XY, YZ y XZ.

    Parameters:
        x_coil_results (DataFrame or ndarray): Datos de los campos magnéticos, como un DataFrame o un arreglo estructurado.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    titles = ['Plano XY', 'Plano YZ', 'Plano XZ']

    for ii in range(0, 3):
        # Magnitud del campo en el plano
        ax1 = axes[ii]

        if ii == 0:
            # Filtro para el plano XY (Z = 0)
            xy_plane = x_coil_results[(x_coil_results['Z'] == 0)]
            x = xy_plane['X'].values
            y = xy_plane['Y'].values
            z = xy_plane['Bx'].values

        elif ii == 1:
            # Filtro para el plano YZ (X = 0)
            yz_plane = x_coil_results[(x_coil_results['X'] == 0)]
            x = yz_plane['Y'].values
            y = yz_plane['Z'].values
            z = yz_plane['Bx'].values

        else:
            # Filtro para el plano XZ (Y = 0)
            xz_plane = x_coil_results[(x_coil_results['Y'] == 0)]
            x = xz_plane['X'].values
            y = xz_plane['Z'].values
            z = xz_plane['Bx'].values

        # Crear la malla 2D de X y Y
        x_grid, y_grid = np.meshgrid(np.unique(x), np.unique(y))

        # Ahora necesitamos reordenar z para que coincida con la malla 2D
        # Crear una malla 2D de los puntos y reorganizar los valores de z en función de esa malla
        z_grid = np.zeros_like(x_grid)  # Crear una matriz de ceros con la misma forma que x_grid

        for i, xi in enumerate(np.unique(x)):
            for j, yi in enumerate(np.unique(y)):
                # Seleccionar el valor correspondiente de z para (xi, yi)
                # Usamos el índice (i, j) para colocar el valor adecuado de z en z_grid
                idx = np.where((x == xi) & (y == yi))[0]
                if len(idx) > 0:
                    z_grid[j, i] = z[idx[0]]  # Asignamos el primer valor de z correspondiente

        # Graficar el contorno
        im = ax1.contourf(x_grid, y_grid, z_grid, cmap='viridis', levels=100, alpha=0.7)
        fig.colorbar(im, ax=ax1)
        ax1.set_title(f"{titles[ii]}: Magnitud del campo")
        ax1.set_xlabel('X' if ii != 1 else 'Y')
        ax1.set_ylabel('Y' if ii == 0 else ('Z' if ii == 1 else 'X'))

    plt.tight_layout()
    plt.show()

def simple_3d_surface_plot(x_coil_results):
    # Define los planos y títulos
    #lanes = ['xy', 'yz', 'xz']
    titles = ['Plano XY', 'Plano YZ', 'Plano XZ']
    
    # Crear un subplot con 1 fila y 3 columnas
    fig = make_subplots(
        rows=1, cols=3, 
        subplot_titles=titles, 
        specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]]
    )

    for ii in range(0, 3):
        # Magnitud del campo en el plano
        if ii == 0:
            # Filtro para el plano XY (Z = 0)
            xy_plane = x_coil_results[(x_coil_results['Z'] == 0)]
            x = xy_plane['X'].values
            y = xy_plane['Y'].values
            z = xy_plane['Bx'].values
            x_label, y_label = "X", "Y"

        elif ii == 1:
            # Filtro para el plano YZ (X = 0)
            yz_plane = x_coil_results[(x_coil_results['X'] == 0)]
            x = yz_plane['Y'].values
            y = yz_plane['Z'].values
            z = yz_plane['Bx'].values
            x_label, y_label = "Y", "Z"

        else:
            # Filtro para el plano XZ (Y = 0)
            xz_plane = x_coil_results[(x_coil_results['Y'] == 0)]
            x = xz_plane['X'].values
            y = xz_plane['Z'].values
            z = xz_plane['Bx'].values
            x_label, y_label = "X", "Z"

        # Crear la malla 2D de X y Y
        x_grid, y_grid = np.meshgrid(np.unique(x), np.unique(y))

        # Ahora necesitamos reordenar z para que coincida con la malla 2D
        # Crear una malla 2D de los puntos y reorganizar los valores de z en función de esa malla
        z_grid = np.zeros_like(x_grid)  # Crear una matriz de ceros con la misma forma que x_grid

        for i, xi in enumerate(np.unique(x)):
            for j, yi in enumerate(np.unique(y)):
                # Seleccionar el valor correspondiente de z para (xi, yi)
                # Usamos el índice (i, j) para colocar el valor adecuado de z en z_grid
                idx = np.where((x == xi) & (y == yi))[0]
                if len(idx) > 0:
                    z_grid[j, i] = z[idx[0]]  # Asignamos el primer valor de z correspondiente

        # Agregar la traza al subplot correspondiente
        fig.add_trace(
            go.Surface(
                z=z_grid,
                x=x_grid,
                y=y_grid,
                colorscale='Viridis',
                showscale=False  # Muestra una sola escala de colores
            ),
            row=1, col=ii + 1
        )

        # Configurar los títulos de los ejes del subplot
        fig.update_scenes(
            dict(
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title="Magnitud"
            ),
            row=1, col=idx + 1
        )

    # Configurar el layout general
    fig.update_layout(
        title="Gráficos de Superficie 3D - Todos los Planos",
        height=700,
        width=1800,  # Ancho ajustado para acomodar 3 gráficos
        showlegend=False,
    )

    # Mostrar la figura con todos los subplots
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
#x_coil_results = pd.read_csv('/home/iaapp/brayan/Helmholtz/x_coil_results.csv')
#plot_magnetic_field(x_coil_results)
#simple_3d_surface_plot(x_coil_results)