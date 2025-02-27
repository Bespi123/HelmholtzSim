import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.dates import DateFormatter

import matplotlib.ticker as ticker
import pandas as pd

Ax = np.eye(3)
Ay = np.array([[0, -1,  0], [1,  0,  0], [0,  0,  1]])
Az = np.array([[0,  0, -1], [0,  1,  0], [1,  0,  0]])

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

def simple_3d_surface_plot(x_coil_results, spires, index='Bx', use_fixed_zaxis=False):
    
    # Variables to store min and max values for automatic z-axis adjustment
    global_min_z = float('inf')
    global_max_z = float('-inf')

    # Define the planes and titles
    titles = [f'{index} in XY-plane', f'{index} in YZ-plane', f'{index} in XZ-plane']
    
    # Check if reference point exists
    reference_point = x_coil_results[(x_coil_results['X'] == 0) & 
                                     (x_coil_results['Y'] == 0) & 
                                     (x_coil_results['Z'] == 0)]
    
    if reference_point.empty:
        reference_value = x_coil_results[index].mean()  # Use mean if no reference found
    else:
        reference_value = reference_point[index].values[0]

    # Calculate the 0.5% tolerance range
    tolerance = 0.005 * reference_value
    lower_bound_tol, upper_bound_tol = reference_value - tolerance, reference_value + tolerance

    # Define the lower and upper bounds
    lower_bound_1 = -1.5 * reference_value
    upper_bound_1 = 1.5 * reference_value

    print('reference_value: ',reference_value)

    # Create a subplot with 1 row and 3 columns
    fig = make_subplots(
        rows=1, cols=3, 
        subplot_titles=titles, 
        specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]]
    )

    for ii in range(0, 3):
        if ii == 0:
            A = Ax
            plane = x_coil_results[x_coil_results['Z'] == 0] 
            x_label, y_label = 'X', 'Y'

        elif ii == 1:
            A = Az
            plane = x_coil_results[x_coil_results['X'] == 0]
            x_label, y_label = 'Y', 'Z'
        elif ii == 2:
            A = Ax
            plane = x_coil_results[x_coil_results['Y'] == 0]
            x_label, y_label = 'X', 'Z'
        else:
            print('Error')
            return
        # Extract data
        x, y, z = plane[x_label].values, plane[y_label].values, plane[index].values
        filtered_points = plane[(plane[index] >= lower_bound_tol) & (plane[index] <= upper_bound_tol)]

        # Generate grid
        x_grid, y_grid = np.meshgrid(np.unique(x), np.unique(y))
        z_grid = np.full_like(x_grid, np.nan, dtype=float)  # Initialize with NaN

        for i, xi in enumerate(np.unique(x)):
            for j, yi in enumerate(np.unique(y)):
                idx = np.where((x == xi) & (y == yi))[0]
                if len(idx) > 0:
                    z_grid[j, i] = np.mean(z[idx])  # Avoid shape mismatch

        # Determine row and col placement
        row, col = 1, ii + 1

        # Add main surface plot
        fig.add_trace(
            go.Surface(
                z=z_grid,
                x=x_grid,
                y=y_grid,
                showscale=True,
                colorscale='Viridis',
                cmin=lower_bound_1/1.5,  # Fix: Dynamic range
                cmax=upper_bound_1/1.5   # Fix: Dynamic range
            ),
            row=row, col=col
        )

        # Ensure filtered_points is not empty before plotting low variation region
        if not filtered_points.empty:
            xx, yy, zz = filtered_points[x_label].values, filtered_points[y_label].values, filtered_points[index].values

            #Debugging: Print values to confirm correctness
            #print(f"Filtered {index} values in {titles[ii]}: {zz}")

            x_grid_1, y_grid_1 = np.meshgrid(np.unique(xx), np.unique(yy))
            z_grid_1 = np.full_like(x_grid_1, np.nan, dtype=float)

            for i, xi in enumerate(np.unique(xx)):
                for j, yi in enumerate(np.unique(yy)):
                    idx = np.where((xx == xi) & (yy == yi))[0]
                    if len(idx) > 0:
                        z_grid_1[j, i] = np.mean(zz[idx])


            # Add Contour or Surface Plot
            fig.add_trace(
                go.Surface(
                    x=x_grid_1,
                    y=y_grid_1,
                    z=z_grid_1,
                    colorscale='Reds',
                    showscale=False,
                    opacity=1,  # Fix: Make it fully visible for testing
                    name="Low Variation (<0.5%) Region"
                ),
                row=row, col=col
            )

         # Apply the flag-controlled z-axis range
        spire11=np.einsum('ij,ljk->lik', A, spires)
        #spire22=np.einsum('ij,ljk->lik', A, spire2)

         # Compute spires' z min/max and update global range
        spire_z_min = np.nanmin([np.nanmin(spire11[..., 2])])
        spire_z_max = np.nanmax([np.nanmax(spire11[..., 2])])

        global_min_z = min(global_min_z, spire_z_min)
        global_max_z = max(global_max_z, spire_z_max)

        zaxis_range = (
            [lower_bound_1, upper_bound_1] if use_fixed_zaxis 
            else [global_min_z, spire_z_max]
        )

        # Update axis labels and z-axis range
        fig.update_scenes(
            dict(
                xaxis_title=f'{x_label} (m)',
                yaxis_title=f'{y_label} (m)',
                zaxis_title="Magnitude",
                 zaxis=dict(range=zaxis_range)
            ),
            row=1, col=ii + 1  
        )

        plot_spires(fig, spire11, color='black', label='X-spires (m)', row=row, col=col)

    fig.update_layout(
        title="Generated Magnetic Field",
        height=700,
        width=1800,
        showlegend=False,
        scene=dict(
            camera=dict(
                eye=dict(x=0, y=0, z=3),  # Vista desde arriba
                up=dict(x=0, y=1, z=0)  # Rotación de 90° en Z
            )
        ),
        scene2=dict(
            camera=dict(
                eye=dict(x=0, y=0, z=3),  # Vista desde arriba
                up=dict(x=0, y=1, z=0)  # Rotación de 90° en Z
            )
        ),
        scene3=dict(
            camera=dict(
                eye=dict(x=0, y=0, z=3),  # Vista desde arriba
                up=dict(x=0, y=1, z=0)  # Rotación de 90° en Z
            )
        )
    )

    fig.show()


def plot_spires(fig, spires, color='black', label='X-spires (m)', row=None, col=None):
    """
    Add spires to an existing Plotly figure. If fig is None, create a new figure.
    
    Automatically detects if fig is a subplot and applies row/col only when needed.

    Parameters:
    - fig (go.Figure or None): Existing Plotly figure. Creates a new one if None.
    - spires (list of numpy.ndarray): List of arrays representing the spires (each with shape (4, 3, 100)).
    - color (str): Color of the spires.
    - label (str): Label for the legend.
    - row (int, optional): Row index for subplots (ignored if fig has no subplots).
    - col (int, optional): Column index for subplots (ignored if fig has no subplots).

    Returns:
    - fig (go.Figure): Updated Plotly figure with spires.
    """
    # If fig is None, create a new figure
    if fig is None:
        fig = go.Figure()

    # Detect if fig has subplots
    has_subplots = hasattr(fig, 'layout') and hasattr(fig.layout, 'grid') and fig.layout.grid is not None

    for i in range(spires.shape[0]):  # Ensure we loop through all spires
        spire_flat=spires[i,:,:].T
    
        # Create Scatter3D trace for each spire
        trace = go.Scatter3d(
            x=spire_flat[:, 0], y=spire_flat[:, 1], z=spire_flat[:, 2],
            mode='lines', line=dict(color=color, width=4),
            name=label if i == 0 else None,  # Show legend only for the first spire
            legendgroup=label, showlegend=(i == 0)  # Group legends to avoid duplicates
        )

        # If fig has subplots, add trace to the correct subplot
        if has_subplots and row is not None and col is not None:
            fig.add_trace(trace, row=row, col=col)
        else:
            fig.add_trace(trace)

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
    if select not in ['ECI', 'ECEF']:
        raise ValueError("Coordinates not supported. Use 'ECI' or 'ECEF'.")

    # Generate Earth data
    earth_x, earth_y, earth_z = create_earth()

     # Seleccionar coordenadas del satélite
    coord_x = df[f"{select} X (km)"].values
    coord_y = df[f"{select} Y (km)"].values
    coord_z = df[f"{select} Z (km)"].values

    # Create a 3D plot
    fig = go.Figure()

    # Add Earth to the plot
    fig.add_trace(go.Surface(
        x=earth_x.reshape(50, 50),
        y=earth_y.reshape(50, 50),
        z=earth_z.reshape(50, 50),
        surfacecolor=np.ones_like(earth_z),  # Color base
        colorscale="Earth", 
        opacity=0.9,
        name="Earth",
        showscale=False,
        lighting=dict(ambient=0.7, diffuse=0.9, specular=0.1, roughness=0.5),  # Ajustar iluminación
        lightposition=dict(x=10000, y=10000, z=10000)
    ))

    # Add satellite trajectory
    fig.add_trace(go.Scatter3d(
        x=coord_x,
        y=coord_y,
        z=coord_z,
        mode='lines',
        line=dict(color='red', width=0.5, dash='dash'),
        name='Satellite Path'
    ))

    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis_title=f"{select} X (km)",
            yaxis_title=f"{select} Y (km)",
            zaxis_title=f"{select} Z (km)",
            aspectmode="data"  # Mantener proporciones iguales
        ),
        title=f"Satellite Trajectory in {select} Coordinates with Earth",
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=40),  # Ajustar márgenes
        legend=dict(x=0.8, y=0.9)  # Posición de la leyenda
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
    # Mapeo de columnas según el sistema de coordenadas
    column_mapping = {
        'ECI': {'Bx': 'Bx ECI (nT)', 'By': 'By ECI (nT)', 'Bz': 'Bz ECI (nT)'},
        'ECEF': {'Bx': 'Bx ECEF (nT)', 'By': 'By ECEF (nT)', 'Bz': 'Bz ECEF (nT)'},
        'NED': {'Bx': 'B N', 'By': 'B E', 'Bz': 'B D'}
    }

    # Verificar si el sistema de coordenadas es válido
    if select not in column_mapping:
        raise ValueError(f"Unsupported coordinate frame: {select}. Use 'ECI', 'ECEF', or 'NED'.")

    # Verificar que las columnas necesarias estén en el DataFrame
    required_columns = [column_mapping[select]['Bx'], column_mapping[select]['By'], column_mapping[select]['Bz'], "Time (UTC)"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # Crear una figura con 3 subplots (uno para cada componente)
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Colores para cada componente
    colors = {'Bx': 'red', 'By': 'blue', 'Bz': 'green'}

    # Graficar cada componente del campo magnético
    for i, (component, column) in enumerate(zip(['Bx', 'By', 'Bz'], [column_mapping[select]['Bx'], column_mapping[select]['By'], column_mapping[select]['Bz']])):
        axs[i].plot(df["Time (UTC)"], df[column], color=colors[component], label=f'{component} ({select})')
        axs[i].set_title(f"Component {component} in {select} coordinates (nT)")
        axs[i].set_ylabel(f"{component} (nT)")
        axs[i].grid(True, alpha=1, linestyle='--', linewidth=0.1, color='gray')  # Personalizar la cuadrícula
        axs[i].legend(loc='upper right')  # Añadir leyenda

    # Formatear el eje X para mostrar fechas de manera más legible
    date_format = DateFormatter("%Y-%m-%d %H:%M")
    axs[-1].xaxis.set_major_formatter(date_format)
    axs[-1].set_xlabel("Time (UTC)")
    plt.xticks(rotation=45)  # Rotar las etiquetas del eje X para mejor legibilidad

    # Reducir el número de ticks en el eje X
    for ax in axs:
        ax.set_xticks(ax.get_xticks()[::10]) 

def plot_2d_magnetic_field(x_coil_results_s, spires, index='Bx', use_fixed_zaxis=True):
    """
    Generates 2D heatmaps with contour lines for magnetic field visualization
    in the XY, YZ, and XZ planes using Matplotlib.
    Also plots the coil spires (spire1, spire2) in each plane.

    Parameters:
    - x_coil_results_s (DataFrame): Magnetic field data (Bx, By, Bz).
    - spire1, spire2 (numpy.ndarray): Arrays representing the spires (shape: (4, 3, N)).
    - index (str): The magnetic field component to visualize ('Bx', 'By', or 'Bz').
    - use_fixed_zaxis (bool): If True, uses a fixed z-axis range; otherwise, it scales automatically.

    Returns:
    - Displays the plots for XY, YZ, and XZ planes with the spires.
    """

    # Compute the reference magnetic field at (X=0, Y=0, Z=0)
    reference_point = x_coil_results_s[
        (x_coil_results_s['X'] == 0) & 
        (x_coil_results_s['Y'] == 0) & 
        (x_coil_results_s['Z'] == 0)
    ]
    
    if reference_point.empty:
        reference_value = x_coil_results_s[index].mean()  # Use mean if no reference found
    else:
        reference_value = reference_point[index].values[0]

    # Calculate the 0.5% tolerance range
    if reference_value == 0:
        tolerance = 0.005
    else:
        tolerance = 0.005 * reference_value
    lower_bound_tol = reference_value - tolerance
    upper_bound_tol = reference_value + tolerance

    # Define the lower and upper bounds
    lower_bound_1 = -1.5 * reference_value
    upper_bound_1 = 1.5 * reference_value

    # Generate multiple contour levels for field variations
    range_values = np.sort(np.array([lower_bound_tol, upper_bound_tol]))
    print('reference_value: ',reference_value)

    # Define the three planes
    planes = [
        ('XY', 'X', 'Y', x_coil_results_s[x_coil_results_s['Z'] == 0]),
        ('YZ', 'Y', 'Z', x_coil_results_s[x_coil_results_s['X'] == 0]),
        ('XZ', 'X', 'Z', x_coil_results_s[x_coil_results_s['Y'] == 0])
    ]

    # Use `constrained_layout=True` to fix spacing issues
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    for ax, (plane_name, x_label, y_label, df) in zip(axes, planes):
        # Create a 2D matrix (pivot table) for the heatmap
        heatmap_data = df.pivot_table(index=x_label, columns=y_label, values=index, aggfunc='mean')

        # Generate coordinate arrays
        x_vals = np.array(heatmap_data.columns, dtype=float)
        y_vals = np.array(heatmap_data.index, dtype=float)
        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        B_field = heatmap_data.values.T

        # Plot main heatmap
        img = ax.imshow(
            B_field, cmap='viridis', origin='lower',
            #extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
            extent=[y_vals.min(), y_vals.max(), x_vals.min(), x_vals.max()],
            vmin=lower_bound_1,  # 🔹 Fixed min color scale
            vmax=upper_bound_1   # 🔹 Fixed max color scale
        )

        # Highlight the tolerance region using `contourf()`
        if reference_value != 0:
            ax.contourf(
                Y, X, B_field,
                levels=[range_values[0], range_values[1]],  # Only fill between tolerance limits
                colors=['red'], alpha=0.4  # Semi-transparent red highlight
            )

            # Overlay standard contour lines
            contours = ax.contour(
                Y, X, B_field, levels=range_values, colors='white', linewidths=1.5
            )

            # Label contour lines
            ax.clabel(
                contours, inline=True, fontsize=10,
                fmt=lambda x: f"{x:.2e} T"
            )

        spire_list = spires.T
        # Transform spires for the current plane
        if plane_name == 'XY':
            spire_x, spire_y = spire_list[:, 0, :], spire_list[:, 1, :]
        elif plane_name == 'YZ':
            spire_x, spire_y = spire_list[:, 1, :], spire_list[:, 2, :]
        elif plane_name == 'XZ':
            spire_x, spire_y = spire_list[:, 0, :], spire_list[:, 2, :]

        # Plot the spires
        ax.plot(spire_x, spire_y, color='black', linestyle='-', linewidth=4, label='spires')

        # Set axis labels and title
        ax.set_xlabel(f"{x_label} (m)")
        ax.set_ylabel(f"{y_label} (m)")
        ax.set_title(f"{index} in {plane_name} plane")

    # Colorbar with proper spacing
    fig.colorbar(img, ax=axes.ravel().tolist(), label=f'{index} (T)')

    # No need for `plt.tight_layout()` anymore!
    plt.legend()
    plt.show()


def plot_mainAxis_field(x_coil_results_s, index='Bx'):
    """
    Plots Bx vs X, Y, and Z for the main axes with a highlighted tolerance region.

    Parameters:
    - x_coil_results_s (DataFrame): Magnetic field data.
    - index (str): Magnetic field component to visualize ('Bx', 'By', 'Bz').

    Returns:
    - A figure with three subplots (Bx vs X, Y, and Z).
    """
    
    # Compute the reference magnetic field at (X=0, Y=0, Z=0)
    reference_point = x_coil_results_s[
        (x_coil_results_s['X'] == 0) & 
        (x_coil_results_s['Y'] == 0) & 
        (x_coil_results_s['Z'] == 0)
    ]
    
    if reference_point.empty:
        reference_value = x_coil_results_s[index].mean()  # Use mean if no reference found
    else:
        reference_value = reference_point[index].values[0]

    # Calculate the 0.5% tolerance range
    tolerance = 0.005 * reference_value if reference_value != 0 else 0.005
    lower_bound_tol, upper_bound_tol = reference_value - tolerance, reference_value + tolerance

    # Define the three axes and filter data accordingly
    lines = [
        ('X', x_coil_results_s[(x_coil_results_s['Y'] == 0) & (x_coil_results_s['Z'] == 0)]),
        ('Y', x_coil_results_s[(x_coil_results_s['X'] == 0) & (x_coil_results_s['Z'] == 0)]),
        ('Z', x_coil_results_s[(x_coil_results_s['X'] == 0) & (x_coil_results_s['Y'] == 0)])
    ]

    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    for ax, (x_label, df) in zip(axes, lines):
        # Filter and sort data
        filtered_points = df[(df[index] >= lower_bound_tol) & (df[index] <= upper_bound_tol)].sort_values(by=x_label)

        # Ensure sorted order for all data
        x_values_full = np.sort(df[x_label].values)
        bx_values_full = df[index].values[np.argsort(df[x_label].values)]

        x_values_filtered = np.sort(filtered_points[x_label].values) if not filtered_points.empty else []
        bx_values_filtered = filtered_points[index].values[np.argsort(filtered_points[x_label].values)] if not filtered_points.empty else []

        # Plot full dataset
        ax.plot(x_values_full, bx_values_full, marker='o', linestyle='-', label='All Data')

        # Overlay the filtered tolerance region
        ax.plot(x_values_filtered, bx_values_filtered, marker='.', linestyle='-', color='red', label='Filtered Range')

        # Set dynamic labels and title
        ax.set_xlabel(f"{x_label} (m)")
        ax.set_ylabel(f"{index} (T)")
        ax.set_title(f"{index} vs {x_label}")
        ax.legend()
        ax.grid(True)

    # Show the figure
    plt.show()


def plot_magnetic_field_directions(x_coil_results_s, spires):
    """
    Generates quiver plots to visualize the magnetic field directions
    using unitary vectors in the XY and XZ planes, with spires plotted.

    Parameters:
    - x_coil_results_s (DataFrame): Magnetic field data.
    - spire1, spire2 (numpy.ndarray): Arrays representing the spires (shape: (4, 3, N)).

    Returns:
    - Displays quiver plots for XY and XZ planes with unitary vectors and spires.
    """

    # Define the two planes with their respective magnetic field components
    planes = [
        ('XY', 'X', 'Y', x_coil_results_s[x_coil_results_s['Z'] == 0], 'Bx', 'By'),
        ('XZ', 'X', 'Z', x_coil_results_s[x_coil_results_s['Y'] == 0], 'Bx', 'Bz')
    ]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)

    for ax, (plane_name, x_label, y_label, df, field_x, field_y) in zip(axes, planes):
        # Create a 2D matrix (pivot table) for the quiver plot
        x_vals = np.sort(df[x_label].unique())
        y_vals = np.sort(df[y_label].unique())
        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

        # Get Bx and By components, replace NaNs with 0
        Bx = df.pivot_table(index=x_label, columns=y_label, values=field_x, aggfunc='mean').values
        By = df.pivot_table(index=x_label, columns=y_label, values=field_y, aggfunc='mean').values
        Bx = np.nan_to_num(Bx)
        By = np.nan_to_num(By)

        # Normalize the vectors to unit length
        magnitude = np.sqrt(Bx**2 + By**2)
        magnitude[magnitude == 0] = 1  # Avoid division by zero
        Bx /= magnitude
        By /= magnitude

        # Add Magnetic Field Direction Arrows (Quiver)
        step = 8  # Reduce arrow density
        ax.quiver(
            X[::step, ::step], Y[::step, ::step], Bx[::step, ::step], By[::step, ::step],
            scale=10, color='red', width=0.005, headwidth=5
        )

        spire_list = spires.T
        # Plot spires for the current plane
        if plane_name == 'XY':
            spire1_x, spire1_y = spire_list[:, 0, :], spire_list[:, 1, :]
        elif plane_name == 'XZ':
            spire1_x, spire1_y = spire_list[:, 0, :], spire_list[:, 2, :]

        # Plot the spires    
        ax.plot(spire1_x[:, :], spire1_y[:, :], color='black', linestyle='-', linewidth=10, label='Spires')

        # Set axis labels and title
        ax.set_xlabel(f"{x_label} (m)")
        ax.set_ylabel(f"{y_label} (m)")
        ax.set_title(f"Magnetic Field Directions in {plane_name} plane")
        ax.grid(True)

    # Add a legend for spires in the last plot
    axes[-1].legend()

    # Show the figure
    plt.show()