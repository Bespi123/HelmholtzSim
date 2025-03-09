# HelmholtzSim: Helmholtz Coils Designer

## Overview

HelmholtzSim is an open-source Python package for simulating and optimizing Helmholtz coil testbeds, enabling the developmen if accurate magnetic testbeds for in-the-loop simulations of ADCS components. It leverages the Biot-Savart law for parallel magnetic field calculations, integrates genetic algorithms for coil optimization, and supports satellite simulation using Two-Line Element (TLE) data. Additionally, it provides 2D and 3D visualization tools. 

HelmholtzSim is developed under a modular approach to perform the following tasks:
- Calculate the required magnetic field range for in-the-loop simulations.
- Simulate the magnetic field inside the magnetic test bed.
- Optimize the design based on the required magnetic field range.
- Calculate the required DC power sources.

## Features

- **Parallel processing customization**
- **Wider range of simulation geometry**
- **Plotting tools**
- **Interactive 3D plotting**
  
## Installation

### Prerequisites

- Python 3.9
- Required Python libraries (install using `requirements.txt`)

### Installation Steps

1. Clone the repository:
   ```sh
   git clone https://github.com/Bespi123/helmholtzCoilsDesigner.git
   cd helmholtzCoilsDesigner
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Example: Computing Magnetic Field

You can use the simulation functions to compute the magnetic field:

```python
import time
import numpy as np
import pandas as pd
from src import helmCoils_simulator as sim
from src import plotMagneticField as hplot

# Initialize coil parameters
number_of_spires = 4
size_length =  [0.9974, 1, 1, 0.9974] 
distance_among_spires = [0.195,0.15,0.195]
turns = [2*30, 30, 30, 2*30]
current = 1 
rotation_matrix = np.eye(3)
X_coil = sim.CoilParameters(number_of_spires, size_length, distance_among_spires, turns, current, rotation_matrix)

# Simulation settings
parallel_coils = 150
batch_Size = 120
grid_length_size = 0.01
num_seg = 100           #Numer of segments

##Spawn spires
spire_x_s = X_coil.square_spires(num_seg)
f0 = None
f0 = hplot.plot_spires(f0, spire_x_s, color='black', row=None, col=None)
f0.show()

#Grid generation
X, Y, Z = sim.generate_range([-0.4, 0.4],[-0.6, 0.6], [-0.6, 0.6], step_size_x = grid_length_size)
hplot.plot_grid(X, Y, Z, f0)

#Run simulations
start_time = time.time() #Count start time
x_coil_results_s = sim.coil_simulation_parallel(X, Y, Z, X_coil, spire_x_s, batch_Size)
end_time = time.time()   #Mark ending time

# Calcular la norma del campo magnético B = sqrt(Bx^2 + By^2 + Bz^2)
x_coil_results_s["B_norm"] = np.sqrt(x_coil_results_s["Bx"]**2 + x_coil_results_s["By"]**2 + x_coil_results_s["Bz"]**2)

#Calculate and show the simulation time
execution_time = end_time - start_time
print(f'Simulation finished in {execution_time/60} minutes...')

# Save results in a CSV file
output_file = 'data/x_four_coil_results.csv'
x_coil_results_s.to_csv(output_file, index=False)
```

## Project Structure

```
helmholtzCoilsDesigner/
│── src/                            # Main source code
|   ├── _init_.py                   # Package initialization
│   ├── helmCoils_optimizer.py      # Genetic Algorithm for coil optimization
│   ├── helmCoils_simulator.py      # Magnetic field computations
│   ├── plotMagneticField.py        # Graphical functions
|   ├── satSimulationMagField.py    # Satellite environment simulation
│── data/                 # Contains data examples
│── notebooks/            # Examples in jupyter notebooks
│── README.md             # Project documentation
│── requirements.txt      # Dependencies
│── main.py               # Entry point for execution
```

## References
- B. Espinoza-Garcia, X. Wang, and P. R. Yanyachi, “Adaptive controller approach for a three-axis Helmholtz magnetic test-bed to test detumbling simulations in the GWSat satellite,” International Journal of Aeronautical and Space Sciences, pp. 1–19, 2025.
- E. H. Cayo, J. P. Contreras, and P. R. Arapa, “Design and implementation of a geomagnetic field simulator for small satellites,” in III IAA Latin American CubeSat Workshop, Ubatuba, Brazil, Jan. 2019, accessed: 2025-02-16. [Online]. Available: \url{https://www.researchgate.net/publication/330116920}
- R. C. da Silva, I. S. K. Ishioka, C. Cappelletti, S. Battistini, and R. A. Borges, “Helmholtz cage design and validation for nanosatellites HWIL testing,” IEEE Transactions on Aerospace and Electronic Systems, vol. 55, no. 6, pp. 3050–3061, 2019.
- A. J. Mäkinen, R. Zetter, J. Iivanainen, K. C. J. Zevenhoven, L. Parkkonen, and R. J. Ilmoniemi, “Magnetic-field modeling with surface currents. Part I: Physical and computational principles of BfieldTools,” Journal of Applied Physics, vol. 128, no. 6, p. 063906, 2020. [Online]. Available: \url{https://doi.org/10.1063/5.0016090}
- R. Zetter, A. J. Mäkinen, J. Iivanainen, K. C. J. Zevenhoven, R. J. Ilmoniemi, and L. Parkkonen, “Magnetic-field modeling with surface currents. Part II: Implementation and usage of BfieldTools,” Journal of Applied Physics, vol. 128, no. 6, p. 063905, 2020. [Online]. Available: \url{https://doi.org/10.1063/5.0016087}
- S. Meng et al., “MagCoilCalc: A Python package for modeling and optimization of axisymmetric magnet coils generating uniform magnetic fields for noble gas spin-polarizers,” SoftwareX, vol. 16, p. 100805, 2021.
- C. Zhao et al., “Design and simulation of a magnetization drive coil based on the Helmholtz principle and an experimental study,” Micromachines, vol. 14, no. 1, 2023. [Online]. Available: \url{https://www.mdpi.com/2072-666X/14/1/152}
- X. Zhu et al., “Optimization of composite Helmholtz coils towards high magnetic uniformity,” Engineering Science and Technology, an International Journal, vol. 47, p. 101539, 2023. [Online]. Available: \url{https://www.sciencedirect.com/science/article/pii/S2215098623002173}
- Q. Cao et al., “Optimization of a coil system for generating uniform magnetic fields inside a cubic magnetic shield,” Energies, vol. 11, no. 3, 2018. [Online]. Available: \url{https://www.mdpi.com/1996-1073/11/3/608}
- K. Wang et al., “Octagonal three-dimensional shim coil structure and design in atomic sensors for magnetic field detection,” IEEE Sensors Journal, vol. 22, no. 6, pp. 5596–5605, 2022.
- J. Uscategui et al., “High-precision magnetic testbed design and simulation for LEO small-satellite control test,” Aerospace, vol. 10, no. 7, 2023. [Online]. Available: \url{https://www.mdpi.com/2226-4310/10/7/640}
- S. D. Sudhoff, Appendix A: Conductor Data and Wire Gauges. John Wiley \& Sons, Ltd, 2021, pp. 589–591. [Online]. Available: \url{https://onlinelibrary.wiley.com/doi/abs/10.1002/9781119674658.app1}
- H. Young and R. Freedman, Sears and Zemansky’s University Physics: With Modern Physics. Vol. 2. New York: Pearson, 2008.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributions

Contributions are welcome! Feel free to submit issues or pull requests.

## Contact

For questions or support, please open an issue on GitHub or contact the project maintainer.

