import numpy as np
import random
from deap import base, creator, tools, algorithms
import src.helmCoils_simulator as sim
import src.plotMagneticField as hplot


# AWG data (remains global)
# AWG (American Wire Gauge) data for various wire sizes, including diameter,
# cross-sectional area, and current-carrying capacity.
# Units:
#   - diameter_mm: Diameter in millimeters (mm)
#   - area_mm2: Cross-sectional area in square millimeters (mm²)
#   - current_A: Current-carrying capacity in amperes (A)
# Source: Table A.2 from 'Sudhoff, S.D. (2021). Appendix A: Conductor Data and Wire 
# Gauges. In Power Magnetic Devices, S.D. Sudhoff (Ed.).
# https://doi.org/10.1002/9781119674658.app1'

awg_data = {
    40: {"diameter_mm": 0.0799, "area_mm2": 0.0031, "current_A": 0.014},
    38: {"diameter_mm": 0.1007, "area_mm2": 0.0049, "current_A": 0.02},
    36: {"diameter_mm": 0.1270, "area_mm2": 0.0079, "current_A": 0.025},
    34: {"diameter_mm": 0.1600, "area_mm2": 0.0127, "current_A": 0.05},
    32: {"diameter_mm": 0.2019, "area_mm2": 0.0201, "current_A": 0.08},
    30: {"diameter_mm": 0.2540, "area_mm2": 0.0317, "current_A": 0.14},
    28: {"diameter_mm": 0.3200, "area_mm2": 0.0501, "current_A": 0.22},
    26: {"diameter_mm": 0.4039, "area_mm2": 0.079, "current_A": 0.36},
    24: {"diameter_mm": 0.5110, "area_mm2": 0.126, "current_A": 0.577},
    22: {"diameter_mm": 0.6438, "area_mm2": 0.205, "current_A": 0.92},
    20: {"diameter_mm": 0.8128, "area_mm2": 0.325, "current_A": 1.46},
    18: {"diameter_mm": 1.0236, "area_mm2": 0.823, "current_A": 2.3},
    16: {"diameter_mm": 1.2908, "area_mm2": 1.31, "current_A": 3.7},
    14: {"diameter_mm": 1.6281, "area_mm2": 2.08, "current_A": 5.9},
    12: {"diameter_mm": 2.0525, "area_mm2": 3.31, "current_A": 9.3},
    10: {"diameter_mm": 2.5883, "area_mm2": 5.26, "current_A": 15.0},
    8: {"diameter_mm": 3.2639, "area_mm2": 8.37, "current_A": 24.0},
    6: {"diameter_mm": 4.1154, "area_mm2": 13.3, "current_A": 37.0},
    4: {"diameter_mm": 5.1894, "area_mm2": 21.2, "current_A": 60.0},
    2: {"diameter_mm": 6.5437, "area_mm2": 33.6, "current_A": 95.0},
    0: {"diameter_mm": 8.2510, "area_mm2": 53.5, "current_A": 150.0},
    -2: {"diameter_mm": 9.2660, "area_mm2": 85.0, "current_A": 200.0},
    -4: {"diameter_mm": 11.684, "area_mm2": 135.0, "current_A": 260.0}
}

# A constant that is truly global and not part of the optimizer's configuration:
# Resistivity of copper in ohm-meters at 20°C (used for calculating wire resistance, etc.).
# Source: Sears and Zemansky's University Physics, Vol. 2, Table 25.1.
RHO = 1.72e-8  # ohm-meters

def resistance_coil(awg_size, N, L):
    """
    Calculate the resistance of a coil made from a given AWG wire size.

    Args:
        awg_size (int): The American Wire Gauge (AWG) size of the wire.
        N (int): The number of turns in the coil.
        L (float): The average length of one turn in meters.

    Returns:
        float: The resistance of the coil in ohms.

    Raises:
        ValueError: If the provided AWG size is not available in the `awg_data` dictionary.

    Notes:
        - The resistivity of copper (RHO) is assumed to be 1.72e-8 ohm-meters.
        - The length of the wire is calculated as L * N, assuming L as the perimeter of the coil.
        - The cross-sectional area of the wire is converted from mm² to m² for unit consistency.
    """
    info = awg_data.get(awg_size)
    if info is None:
        raise ValueError("AWG gauge not available.")
    length = L * N
    area = info['area_mm2'] * 1e-6
    return RHO * (length / area)

def calculate_loop_length(coordinates):
    """
    Calculates the length of each loop in an array of shape (coil_number,3,N).
    
    Parameters:
        coordinates (numpy.ndarray): Array of coordinates with shape (coil_number,3,N)
    
    Returns:
        list: List with the lengths of each loop
    """
    num_loops = coordinates.shape[0]
    lengths = []
    
    for i in range(num_loops):
        # Get coordinates of loop i
        x, y, z = coordinates[i]
        
        # Compute differences between consecutive points
        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        
        # Compute Euclidean distance between consecutive points and sum them
        length = np.sum(np.sqrt(dx**2 + dy**2 + dz**2))
        lengths.append(length)
        
    return lengths

def select_awg(current, awg_data):
    """
    Selects the smallest AWG that can handle at least the given current.
    
    Parameters:
        current (float): The required current in amperes (A).
        awg_data (dict): Dictionary containing AWG data.
    
    Returns:
        int: The closest AWG number that meets or exceeds the current requirement.
    """
    # Filter AWG values that can handle at least the required current
    valid_awgs = [awg for awg in awg_data if awg_data[awg]["current_A"] >= current]
    
    if not valid_awgs:
        return None  # No valid AWG found
    
    # Select the AWG with the smallest diameter (largest AWG number)
    best_awg = min(valid_awgs, key=lambda awg: awg_data[awg]["current_A"])
    
    return best_awg

def get_slope(coil, spires):
    """
    Calculate the slope of the magnetic field (B_x) with respect to the current (I) for a given coil configuration.

    Args:
        coil: The coil object, which can be updated with parameters like turns and current.
        spires: The number of spires (turns) in the coil.

    Returns:
        float: The slope of the linear fit between current (I) and the magnetic field (B_x).

    Notes:
        - The function simulates the magnetic field (B_x) for different current values (I) and fits a linear model to the data.
        - The slope represents the relationship between the current and the magnetic field, which is useful for estimating the coil's sensitivity.
        - The simulation is performed in parallel for efficiency, and progress bars are disabled to reduce output clutter.
    """
    # Initialize simulation data
    data = []

    # Generate a range of points in space for the simulation
    # Here, X, Y, Z are all set to [0, 0], meaning the simulation is focused on the origin.
    X, Y, Z = sim.generate_range([0, 0], [0, 0], [0, 0], step_size_x=0.01)

    # Compute slope for B_x estimation by varying the current (I)
    for I in range(1, 5):
        # Update the coil parameters with the current value and a fixed number of turns
        coil.update_parameters(turns=1, current=I)
        # Run the coil simulation in parallel
        results = sim.coil_simulation_parallel(
            X, Y, Z, coil, spires, batch_size = 120, enable_progress_bar=False, n=150
        )
        # Store the current (I) and the corresponding B_x value at the first point
        data.append([I, results['Bx'][0]])

    # Convert the data to a NumPy array for easier manipulation
    data = np.array(data)

    # Perform a linear fit to the data: B_x = slope * I + intercept
    # The slope represents the sensitivity of the magnetic field to the current
    slope, intercept = np.polyfit(data[:, 0], np.sum(coil.N) * data[:, 1], 1)

    # Return the slope
    return slope    

# Define the optimizer as a class:
class Source_optimizer:
    def __init__(self, desired_magField, coil, spires, fixed_V_limit=None, max_N = 30,
                 max_I = 10, population = 20, generations = 50, mutation = 0.2):
        """
        Parameters:
          desired_size: base size parameter for coil dimensions.
          spires_function: function to generate coil geometry.
          N: Number of turns.
          I: Current.
          fix_L: Boolean flag; if True, L will remain fixed.
          fixed_L_value: If fix_L is True, the fixed value for L.
          grid_length_size: Parameter used in fitness evaluation.
        """
        self.desired_magField = desired_magField
        self.coil = coil
        self.spires = spires
        self.V_limit = fixed_V_limit
        self.max_N  = max_N
        self.max_I  = max_I
        self.pop = population
        self.gen = generations
        self.mut = mutation

        # Spires perimeter and get slope
        self.perimeter = np.sum(calculate_loop_length(spires))
        self.slope = get_slope(coil, spires)

        # Set ranges for N and V based on fix_V flag.
        self.min_I = 0.001
        self.min_N = 1

        # A local cache for fitness evaluations
        self.fitness_cache = {}

        # Set up DEAP toolbox.
        self._setup_deap()

    def _setup_deap(self):
        # Create DEAP types if not already created.
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()

        # Register genetic operators
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.init_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", lambda ind: self.fitness_function(ind)) #lambda used to add new parameters
        self.toolbox.register("mate", self.long_jump_crossover)
        self.toolbox.register("mutate", self.mutate_individual, mu=0, sigma=0.1, indpb=0.4)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        # Register attribute generators
        self.toolbox.register("attr_I", random.uniform, self.min_I, self.max_I)
        self.toolbox.register("attr_N", random.uniform, self.min_N, self.max_N)

    def apply_constraints(self, individual):
        individual[0] = round(max(self.min_I, min(self.max_I, individual[0])), 2)
        individual[1] = round(max(self.min_N, min(self.max_N, individual[1])), 0)
        return individual

    def init_individual(self):
        # Wider range for initialization
        initial_I = random.uniform(self.min_I * 0.5, self.max_I * 1.5)  # Extends beyond normal range
        initial_N = random.uniform(self.min_N * 0.5, self.max_N * 1.5)  # Allows more extreme values

        # Ensure the values stay within constraints
        ind = creator.Individual([initial_I, initial_N])
        return self.apply_constraints(ind)


    def fitness_function(self, individual):
        """
        Evaluates the fitness of an individual (I, N) in the genetic algorithm.

        The goal is to minimize the power consumption while ensuring that the generated 
        magnetic field meets or exceeds the desired value, and that the voltage does not exceed 
        the given limit.

        Parameters:
            individual (list): A list containing two elements:
                - I (float): Current in amperes.
                - N (float): Number of turns in the coil.

        Returns:
            tuple: A single-element tuple containing the fitness value, where lower is better.
        """

        # Extract individual parameters
        I, N = individual

        # Use a cache to avoid redundant fitness evaluations
        key = (I, N)

        if key in self.fitness_cache:
            return self.fitness_cache[key]

        # Access the coil object stored in the class
        coil = self.coil
        coil.update_parameters(turns=N)
        # Compute total number of turns
        N_total = np.sum(coil.N)

        # Select appropriate AWG size based on the current requirement
        selected_awg = select_awg(I, awg_data)

        # Compute wire resistance based on selected AWG, total turns, and perimeter
        R = resistance_coil(selected_awg, N_total, self.perimeter)
     
        # Compute voltage drop across the coil
        V = I * R

        # Compute the estimated magnetic field strength
        B_x = self.slope * N_total * I

        # Target magnetic field strength
        target = self.desired_magField

        # Apply penalty if the generated field is below the target
        if (target - B_x) > 0:
            penalty1 = 50000 + abs(target - B_x)  # Higher penalty for larger deviation
        else:
            penalty1 = 0

        # Apply penalty if the voltage exceeds the limit
        if (self.V_limit - V) < 0:
            penalty2 = 50000 + abs(self.V_limit - V)  # Higher penalty for exceeding constraints
        else:
            penalty2 = 0

        # Compute power consumption
        power = V * I

        # Total fitness value: penalized power consumption
        result = (penalty1 + penalty2 + power + I,)

        # Cache the computed fitness value to speed up future evaluations
        self.fitness_cache[key] = result

        return result

    def mutate_individual(self, individual, mu, sigma, indpb):
        if random.random() < indpb:
            individual[0] += random.gauss(mu, sigma)
        if random.random() < indpb:
            individual[1] += random.gauss(mu, sigma)
        return self.apply_constraints(individual),

    def adaptive_mutate(self, individual, gen, mu):
        """Mutación adaptativa con mayor exploración al inicio."""
        mutation_rate = 0.5 * (1 - gen / self.gen)
        sigma = 0.2 * (1 - gen / self.gen)
        
        if random.random() < mutation_rate:
            individual[0] += random.gauss(mu, sigma)

        if random.random() < mutation_rate:
            individual[1] += random.gauss(mu, sigma)

        return self.apply_constraints(individual),

    def mate_individual(self, ind1, ind2): 
        'With 50% of probability generates individuals'
        if not self.fix_L:
            if random.random() < 0.5:
                ind1[0], ind2[0] = ind2[0], ind1[0]
        if random.random() < 0.5:
            ind1[1], ind2[1] = ind2[1], ind1[1]
        self.apply_constraints(ind1)
        self.apply_constraints(ind2)
        return ind1, ind2

    def long_jump_crossover(self, ind1, ind2):
        """Cruce con exploración agresiva con mejor probabilidad de mezcla."""
        
        # Swapping genes with 25% probability
        if random.random() < 0.25:
            ind1[0], ind2[0] = ind2[0], ind1[0]
        if random.random() < 0.25:
            ind1[1], ind2[1] = ind2[1], ind1[1]

        # Blended crossover with 50% probability
        if random.random() < 0.5:
            alpha = random.uniform(-0.5, 1.5)
            ind1[0] = alpha * ind1[0] + (1 - alpha) * ind2[0]
            ind2[0] = alpha * ind2[0] + (1 - alpha) * ind1[0]
        
        if random.random() < 0.5:
            alpha = random.uniform(-0.5, 1.5)
            ind1[1] = alpha * ind1[1] + (1 - alpha) * ind2[1]
            ind2[1] = alpha * ind2[1] + (1 - alpha) * ind1[1]

        self.apply_constraints(ind1)
        self.apply_constraints(ind2)
    
        return ind1, ind2
    
    def run_ga(self, pop_size=None, cxpb=0.5, mutpb=None, ngen=None, initial_individual=None):
        if pop_size is None:
            pop_size = self.pop
        if ngen is None:
            ngen = self.gen
        if mutpb is None:
            mutpb = self.mut

        # Generate initial population (reserve space for one extra individual)
        pop = self.toolbox.population(n=pop_size - 1)

        # Add the initial individual if provided
        if initial_individual:
            ind = creator.Individual(initial_individual)
            ind.fitness.values = self.toolbox.evaluate(ind)  # Evaluate fitness
            pop.append(ind)  # Insert into population

        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)

        pop, logbook = algorithms.eaSimple(pop, self.toolbox, cxpb=cxpb, mutpb=mutpb,
                                        ngen=ngen, stats=stats, halloffame=hof, verbose=True)
        return hof[0], logbook


    def optimize(self):
        best_solution, logbook = self.run_ga(initial_individual=[1.05, 0.59])
        I_opt, N_opt = best_solution
        print("\nOptimal Parameters Found:")
        print(f"I (A): {I_opt:.4f} A")
        print(f"N (-): {N_opt:.4f}")

        selected_awg = select_awg(I_opt, awg_data)
        R = resistance_coil(selected_awg, N_opt, self.perimeter)
        V = I_opt*R
        print(f"coil resistnace (ohm): {R:.4f} ohm")
        print(f"V (v): {V:.4f} v")
        return I_opt, N_opt
    

# Define the optimizer as a class:
class HelmholtzOptimizer:
    def __init__(self, desired_size, coil, fun, fix_L=False, fixed_L_value=None,
                 grid_length_size=0.01, population = 20, generations = 50, mutation = 0.2):
        """
        Parameters:
          desired_size: base size parameter for coil dimensions.
          spires_function: function to generate coil geometry.
          N: Number of turns.
          I: Current.
          fix_L: Boolean flag; if True, L will remain fixed.
          fixed_L_value: If fix_L is True, the fixed value for L.
          grid_length_size: Parameter used in fitness evaluation.
        """
        self.desired_size = desired_size
        self.coil = coil
        self.fun = fun
        self.fix_L = fix_L
        self.fixed_L_value = fixed_L_value
        self.grid_length_size = grid_length_size
        self.pop = population
        self.gen = generations
        self.mut = mutation
        # Set ranges for L and d based on fix_L flag.
        if self.fix_L:
            if self.fixed_L_value is None:
                raise ValueError("When fix_L is True, fixed_L_value must be provided")
            self.min_L, self.max_L = self.fixed_L_value, self.fixed_L_value
            self.min_d, self.max_d = self.fixed_L_value / 2, self.fixed_L_value
        else:
            self.min_L, self.max_L = self.desired_size, self.desired_size * 4
            self.min_d, self.max_d = self.desired_size, self.desired_size * coil.coils_number

        # A local cache for fitness evaluations
        self.fitness_cache = {}

        # Set up DEAP toolbox.
        self._setup_deap()

    def _setup_deap(self):
        # Create DEAP types if not already created.
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()

        # Register genetic operators
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.init_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", lambda ind: self.fitness_function(ind,batch_Size = 120, num_seg=100)) #lambda used to add new parameters
        #self.toolbox.register("mate", self.mate_individual)  #Combine genes of two generations
        self.toolbox.register("mate", self.long_jump_crossover)
        self.toolbox.register("mutate", self.mutate_individual, mu=0, sigma=0.1, indpb=0.4)
        #self.toolbox.register("mutate", self.adaptive_mutate, mu=0, sigma=0.1, indpb=0.4)
        self.toolbox.register("select", tools.selTournament, tournsize=3)


        # Register attribute generators
        self.toolbox.register("attr_L", random.uniform, self.min_L, self.max_L)
        self.toolbox.register("attr_d", random.uniform, self.min_d, self.max_d)

    def apply_constraints(self, individual):
        # If fix_L is True, force L to the fixed value.
        if self.fix_L:
            individual[0] = self.fixed_L_value
        else:
            individual[0] = round(max(self.min_L, min(self.max_L, individual[0])), 2)
        individual[1] = round(max(self.min_d, min(self.max_d, individual[1])), 2)
        return individual

    def init_individual(self):
        # Wider range for initialization
        initial_L = random.uniform(self.min_L * 0.5, self.max_L * 1.5)  # Extends beyond normal range
        initial_d = random.uniform(self.min_d * 0.5, self.max_d * 1.5)  # Allows more extreme values

        # Ensure the values stay within constraints
        ind = creator.Individual([initial_L, initial_d])
        return self.apply_constraints(ind)


    def fitness_function(self, individual, grid_length_size = 0.01, batch_Size = 120, *args, **kwargs):
        L, d = individual
        key = (L, d)
        if key in self.fitness_cache:
            return self.fitness_cache[key]

        coil = self.coil
        # Update the coil parameters
        coil.update_parameters(length=L, distance=d)

        spires = self.fun(*args, **kwargs)

        X, Y, Z = sim.generate_range([-1*(np.sum(coil.h)/2), 0], step_size_x = grid_length_size)

        coil_Results = sim.coil_simulation_parallel(
            X, Y, Z, coil, spires, batch_Size, enable_progress_bar=False
        )

        #hplot.plot_mainAxis_field(coil_Results, index='Bx')

        target = coil_Results[(coil_Results['X'] == 0) & (coil_Results['Y'] == 0) & (coil_Results['Z'] == 0)]
        if target.empty:
            target_point = coil_Results['Bx'].mean()
        else:
            target_point = target['Bx'].values[0]
        tolerance = 0.001 * target_point if target_point != 0 else 0.001
        lower_bound_tol, upper_bound_tol = target_point - tolerance, target_point + tolerance

        filtered_points = coil_Results[
            (coil_Results['Bx'] >= lower_bound_tol) & (coil_Results['Bx'] <= upper_bound_tol)
        ].sort_values(by='X')

        if len(filtered_points) > 1:
            is_contiguous = all(
                (filtered_points['X'].iloc[i + 1] - filtered_points['X'].iloc[i]) <= 2 * self.grid_length_size
                for i in range(len(filtered_points) - 1)
            )
        else:
            is_contiguous = False

        if not is_contiguous:
            result = (5000,)  # Penalty value
        else:
            a = abs(filtered_points['X'].max() - filtered_points['X'].min())
            result = (self.desired_size / 2 - a,)

        self.fitness_cache[key] = result
        return result

    def mutate_individual(self, individual, mu, sigma, indpb):
        if not self.fix_L:
            if random.random() < indpb:
                individual[0] += random.gauss(mu, sigma)
        if random.random() < indpb:
            individual[1] += random.gauss(mu, sigma)
        return self.apply_constraints(individual),

    def adaptive_mutate(self, individual, gen, mu):
        """Mutación adaptativa con mayor exploración al inicio."""
        mutation_rate = 0.5 * (1 - gen / self.gen)
        sigma = 0.2 * (1 - gen / self.gen)
        
        if not self.fix_L:
            if random.random() < mutation_rate:
                individual[0] += random.gauss(mu, sigma)

        if random.random() < mutation_rate:
            individual[1] += random.gauss(mu, sigma)

        return self.apply_constraints(individual),

    def mate_individual(self, ind1, ind2): 
        'With 50% of probability generates individuals'
        if not self.fix_L:
            if random.random() < 0.5:
                ind1[0], ind2[0] = ind2[0], ind1[0]
        if random.random() < 0.5:
            ind1[1], ind2[1] = ind2[1], ind1[1]
        self.apply_constraints(ind1)
        self.apply_constraints(ind2)
        return ind1, ind2

    def long_jump_crossover(self, ind1, ind2):
        """Cruce con exploración agresiva."""
        alpha = random.uniform(-0.5, 1.5)

        if not self.fix_L:
            ind1[0], ind2[0] = ind2[0], ind1[0]
        
        ind1[1] = alpha * ind1[1] + (1 - alpha) * ind2[1]
        self.apply_constraints(ind1)
        self.apply_constraints(ind2)
        return ind1, ind2
    
    def run_ga(self, pop_size=None, cxpb=0.5, mutpb=None, ngen=None, initial_individual=None):
        if pop_size is None:
            pop_size = self.pop
        if ngen is None:
            ngen = self.gen
        if mutpb is None:
            mutpb = self.mut

        # Generate initial population (reserve space for one extra individual)
        pop = self.toolbox.population(n=pop_size - 1)

        # Add the initial individual if provided
        if initial_individual:
            ind = creator.Individual(initial_individual)
            ind.fitness.values = self.toolbox.evaluate(ind)  # Evaluate fitness
            pop.append(ind)  # Insert into population

        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)

        pop, logbook = algorithms.eaSimple(pop, self.toolbox, cxpb=cxpb, mutpb=mutpb,
                                        ngen=ngen, stats=stats, halloffame=hof, verbose=True)
        return hof[0], logbook


    def optimize(self):
        best_solution, logbook = self.run_ga(initial_individual=[1.05, 0.59])
        L_opt, d_opt = best_solution
        print("\nOptimal Parameters Found:")
        print(f"L (length): {L_opt:.4f} m")
        print(f"d (spacing): {d_opt:.4f} m")
        return L_opt, d_opt