import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D

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

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def simple_3d_surface_plot(x_coil_results, index='Bx'):
    # Define the planes and titles
    titles = [f'{index} XY-plane', f'{index} YZ-plane', f'{index} XZ-plane']
    
    # Create a subplot with 1 row and 3 columns
    fig = make_subplots(
        rows=1, cols=3, 
        subplot_titles=titles, 
        specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]]
    )

    for ii in range(0, 3):
        if ii == 0:
            xy_plane = x_coil_results[(x_coil_results['Z'] == 0)]
            x, y, z = xy_plane['X'].values, xy_plane['Y'].values, xy_plane[index].values
            x_label, y_label = "X", "Y"
        elif ii == 1:
            yz_plane = x_coil_results[(x_coil_results['X'] == 0)]
            x, y, z = yz_plane['Y'].values, yz_plane['Z'].values, yz_plane[index].values
            x_label, y_label = "Y", "Z"
        else:
            xz_plane = x_coil_results[(x_coil_results['Y'] == 0)]
            x, y, z = xz_plane['X'].values, xz_plane['Z'].values, xz_plane[index].values
            x_label, y_label = "X", "Z"

        x_grid, y_grid = np.meshgrid(np.unique(x), np.unique(y))
        z_grid = np.zeros_like(x_grid)

        for i, xi in enumerate(np.unique(x)):
            for j, yi in enumerate(np.unique(y)):
                idx = np.where((x == xi) & (y == yi))[0]
                if len(idx) > 0:
                    z_grid[j, i] = np.mean(z[idx])

        # Determine row and col placement
        row, col = 1, ii + 1

        fig.add_trace(
            go.Surface(
                z=z_grid,
                x=x_grid,
                y=y_grid,
                colorscale='Viridis',
                showscale=False
            ),
            row=row, col=col
        )

        # ADD CONTOUR PLOT FOR LOW VARIATION REGION (YZ PLANE)
        if index == 'B_norm' and ii == 1:
            # Define the reference point at (X=0, Y=0, Z=0)
            reference_point = x_coil_results[(x_coil_results['X'] == 0) & 
                                             (x_coil_results['Y'] == 0) & 
                                             (x_coil_results['Z'] == 0)]

            # Get the reference value of B_norm
            reference_value = reference_point['B_norm'].values[0] if not reference_point.empty else x_coil_results['B_norm'].mean()

            # Calculate the 0.5% tolerance range
            tolerance = 0.005 * reference_value
            lower_bound, upper_bound = reference_value - tolerance, reference_value + tolerance

            # Filter points that meet the condition in the YZ plane (X=0)
            yz_plane = x_coil_results[x_coil_results['X'] == 0]
            filtered_points = yz_plane[(yz_plane['B_norm'] >= lower_bound) & (yz_plane['B_norm'] <= upper_bound)]

            # Ensure filtered_points is not empty before plotting
            if not filtered_points.empty:
                x_grid, y_grid = np.meshgrid(np.unique(filtered_points['Y']), np.unique(filtered_points['Z']))
                z_grid = np.full_like(x_grid, np.nan, dtype=float)
                #z_grid = np.zeros_like(x_grid)

                for i, xi in enumerate(np.unique(filtered_points['Y'])):
                    for j, yi in enumerate(np.unique(filtered_points['Z'])):
                        idx = np.where((filtered_points['Y'] == xi) & (filtered_points['Z'] == yi))[0]
                        if len(idx) > 0:
                            z_grid[j, i] = filtered_points['B_norm'].values[idx]

                # Add Contour or Surface Plot
                fig.add_trace(
                    go.Surface(
                        x=x_grid,
                        y=y_grid,
                        z=z_grid,
                        colorscale='Reds',
                        showscale=False,
                        opacity=1,
                        name="Low Variation (<0.5%) Region"
                    ),
                    row=row, col=col  # Ensure it only appears in the YZ-plane
                )

        fig.update_scenes(
            dict(
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title="Magnitude"
            ),
            row=1, col=ii + 1  
        )

    fig.update_layout(
        title="Generated Magnetic Field",
        height=700,
        width=1800,
        showlegend=False,
    )

    fig.show()



def plot_spires(fig, spire1, spire2, color='black', label='X-spires'):
    """
    Add spires to an existing Plotly figure or create a new one if `fig` is None.
    
    Parameters:
    - fig (go.Figure or None): Existing Plotly figure to update. Creates a new one if None.
    - spire1, spire2 (numpy.ndarray): Arrays representing the spires (shape: (4, 3, 100)).
    - color (str): Color of the spires.

    Returns:
    - fig (go.Figure): Updated Plotly figure.
    """
    # Change the shape of spire1 and spire2 from (4, 3, 100) to (4, 100, 3)
    spire1 = spire1.transpose(0, 2, 1)  # Shape: (4, 100, 3)
    spire2 = spire2.transpose(0, 2, 1)  # Shape: (4, 100, 3)
    
    # Flatten the arrays to (num_points, 3) per spire
    spire1_flat = spire1.reshape(-1, 3)  # Flatten to (400, 3)
    spire2_flat = spire2.reshape(-1, 3)  # Flatten to (400, 3)
    
    # Create a new figure if `fig` is None
    if fig is None:
        fig = go.Figure()
    
    # Add traces for spire1 and spire2
    fig.add_trace(go.Scatter3d(
        x=spire1_flat[:, 0], 
        y=spire1_flat[:, 1], 
        z=spire1_flat[:, 2], 
        mode='lines', 
        line=dict(color=color, width=4),
        name=label,  # Shared legend label
        legendgroup=label,  # Groups the legend entries
        showlegend=True  # Only show legend once
    ))
    
    fig.add_trace(go.Scatter3d(
        x=spire2_flat[:, 0], 
        y=spire2_flat[:, 1], 
        z=spire2_flat[:, 2], 
        mode='lines', 
        line=dict(color=color, width=4),
        name=label,  # Shared legend label
        legendgroup=label,  # Groups the legend entries
        showlegend=False    # Only show legend once
    ))

    return fig


def plot_grid(X, Y, Z, fig):
  
  points = np.stack((X, Y, Z), axis=-1)  # concatenate coordinates

  # 1) Definir las máscaras
  mask_xy = (points[..., 2] == 0)  # plano XY: Z=0 
  mask_yz = (points[..., 0] == 0)  # plano YZ: X=0
  mask_xz = (points[..., 1] == 0)  # plano XZ: Y=0

  # 2) Filtrar
  P1_points = points[mask_xy]  # array con shape (N1, 3)
  P2_points = points[mask_yz]  # array con shape (N2, 3)
  P3_points = points[mask_xz]  # array con shape (N3, 3)

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

# Function to create a sphere representing the Earth
def create_earth(radius=6371.0, resolution=50):
    # Sphere generation
    phi = np.linspace(0, 2 * np.pi, resolution)
    theta = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(phi), np.sin(theta))
    y = radius * np.outer(np.sin(phi), np.sin(theta))
    z = radius * np.outer(np.ones_like(phi), np.cos(theta))
    return x.flatten(), y.flatten(), z.flatten()

def plot_orbit(df, select):
    # Generate Earth data
    earth_x, earth_y, earth_z = create_earth()

    if select == 'ECI':
        # Satellite trajectory data
        coord_x = df["ECI X (km)"].values
        coord_y = df["ECI Y (km)"].values
        coord_z = df["ECI Z (km)"].values
        label = 'ECI'
    elif select == 'ECEF':
        coord_x = df["ECEF X (km)"].values
        coord_y = df["ECEF Y (km)"].values
        coord_z = df["ECEF Z (km)"].values
        label = 'ECEF'
    else:
        print('Coordinates not supported.')

    # Create a 3D plot
    fig = go.Figure()

    # Add Earth to the plot
    fig.add_trace(go.Surface(
        x=earth_x.reshape(50, 50),
        y=earth_y.reshape(50, 50),
        z=earth_z.reshape(50, 50),
        colorscale="Blues",
        opacity=0.5,
        name="Earth",
        showscale=False
    ))

    # Add satellite trajectory
    fig.add_trace(go.Scatter3d(
        x=coord_x,
        y=coord_y,
        z=coord_z,
        mode='lines',
        line=dict(color='red', width=4),
        name='Satellite Path'
    ))

    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
        xaxis_title="X (km)",
        yaxis_title="Y (km)",
        zaxis_title="Z (km)",
        aspectmode="data"  # Equal aspect ratio
    ),
    title = f"Satellite Trajectory in {label} Coordinates with Earth",
    showlegend=True
    )

    # Show the plot
    fig.show()

def plot_magField_time(df, select):
    """
    Plot the magnetic field components over time for a given coordinate system.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing magnetic field data.
    - select (str): Coordinate system ('ECI', 'ECEF', or 'NED').

    Returns:
    - None: Displays the plot.
    """
    # Create a figure with 3 subplots (one for each component)
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    if select == 'ECI':
        Bx_column = 'Bx ECI (nT)'
        By_column = 'By ECI (nT)'
        Bz_column = 'Bz ECI (nT)'
        label = 'ECI'
    elif select == 'ECEF':
        Bx_column = 'Bx ECEF (nT)'
        By_column = 'By ECEF (nT)'
        Bz_column = 'Bz ECEF (nT)'
        label = 'ECEF'
    elif select == 'NED':
        Bx_column = 'B N'
        By_column = 'B E'
        Bz_column = 'B D'
        label = 'NED'
    else:
        print('Not supported coordinate frame.')
        return  # Exit the function if an unsupported coordinate frame is provided

    # Plot each magnetic field component
    axs[0].plot(df["Time (UTC)"], df[Bx_column], color="red")
    axs[0].set_title(f"Component Bx in {label} coordinates (nT)")
    axs[0].set_ylabel(Bx_column)

    axs[1].plot(df["Time (UTC)"], df[By_column], color="blue")
    axs[1].set_title(f"Component By in {label} coordinates (nT)")
    axs[1].set_ylabel(By_column)

    axs[2].plot(df["Time (UTC)"], df[Bz_column], color="green")
    axs[2].set_title(f"Component Bz in {label} coordinates (nT)")
    axs[2].set_ylabel(Bz_column)
    axs[2].set_xlabel("Time (UTC)")

    # Reduce the number of ticks on the x-axis
    for ax in axs:
        ax.tick_params(axis='x', which='both', labelrotation=45)
        ax.set_xticks(ax.get_xticks()[::10])  # Show only every 10th tick

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.show()
