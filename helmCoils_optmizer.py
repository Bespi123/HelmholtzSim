import numpy as np
import random
from deap import base, creator, tools, algorithms
import parallel_test4 as test_3

# AWG data (remains global if it is constant)
# Constants
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
RHO = 1.68e-8  # Resistivity of copper in ohm-meters

# Optionally, a helper function if needed
def resistance_coil(awg_size, N, L):
    info = awg_data.get(awg_size)
    if info is None:
        raise ValueError("AWG gauge not available.")
    length = 4 * L * N
    area = info['area_mm2'] * 1e-6
    return RHO * (length / area)

# Define the optimizer as a class:
class HelmholtzOptimizer:
    def __init__(self, desired_size, spires_function, Ax, X, Y, Z, N=30, I=1, fix_L=False, fixed_L_value=None,
                 grid_length_size=0.01, population = 20, generations = 50):
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
        self.spires_function = spires_function
        self.Ax = Ax
        self.X = X
        self.Y = Y 
        self.Z = Z
        self.N = N
        self.I = I
        self.fix_L = fix_L
        self.fixed_L_value = fixed_L_value
        self.grid_length_size = grid_length_size
        self.pop = population
        self.gen = generations

        # Set ranges for L and d based on fix_L flag.
        if self.fix_L:
            if self.fixed_L_value is None:
                raise ValueError("When fix_L is True, fixed_L_value must be provided")
            self.min_L, self.max_L = self.fixed_L_value, self.fixed_L_value
            self.min_d, self.max_d = self.fixed_L_value / 2, self.fixed_L_value
        else:
            self.min_L, self.max_L = self.desired_size, self.desired_size * 4
            self.min_d, self.max_d = self.desired_size, self.desired_size * 2

        # A local cache for fitness evaluations
        self.fitness_cache = {}

        # Set up DEAP toolbox.
        self.toolbox = base.Toolbox()
        self._setup_deap()

    def _setup_deap(self):
        # Create DEAP types if not already created.
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        # Register attribute generators
        self.toolbox.register("attr_L", random.uniform, self.min_L, self.max_L)
        self.toolbox.register("attr_d", random.uniform, self.min_d, self.max_d)
        self.toolbox.register("individual", self.init_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Register genetic operators
        self.toolbox.register("mutate", self.mutate_individual, mu=0, sigma=0.1, indpb=0.4)
        self.toolbox.register("mate", self.mate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        # Evaluation function: pass self.spires_function and other parameters
        self.toolbox.register("evaluate", lambda ind: self.fitness_function(ind, self.N, self.I, self.spires_function))

    def apply_constraints(self, individual):
        # If fix_L is True, force L to the fixed value.
        if self.fix_L:
            individual[0] = self.fixed_L_value
        else:
            individual[0] = round(max(self.min_L, min(self.max_L, individual[0])), 2)
        individual[1] = round(max(self.min_d, min(self.max_d, individual[1])), 2)
        return individual

    def init_individual(self):
        ind = creator.Individual([self.toolbox.attr_L(), self.toolbox.attr_d()])
        return self.apply_constraints(ind)

    def fitness_function(self, individual, N, I, square_geometry, num_seg=100):
        L, d = individual
        key = (L, d)
        if key in self.fitness_cache:
            return self.fitness_cache[key]

        # Simulation using test_3 functions; Ax, X, Y, Z should be defined in your context.
        coil = test_3.CoilParameters(L, d, N)
        spire1, spire2 = square_geometry(self.Ax, coil.h, coil.a, num_seg)
        coil_Results = test_3.coil_simulation_1d_sequential(
            self.X, self.Y, self.Z, coil, I, spire1, spire2, 1, 20, False
        )
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

    def mate_individual(self, ind1, ind2):
        if not self.fix_L:
            if random.random() < 0.5:
                ind1[0], ind2[0] = ind2[0], ind1[0]
        if random.random() < 0.5:
            ind1[1], ind2[1] = ind2[1], ind1[1]
        self.apply_constraints(ind1)
        self.apply_constraints(ind2)
        return ind1, ind2

    def run_ga(self, pop_size = None, cxpb=0.5, mutpb=0.2, ngen=None):
        if pop_size is None:
            pop_size = self.pop
        if ngen is None:
            ngen = self.gen
        pop = self.toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)
        pop, logbook = algorithms.eaSimple(pop, self.toolbox, cxpb=cxpb, mutpb=mutpb,
                                           ngen=ngen, stats=stats, halloffame=hof, verbose=True)
        return hof[0], logbook

    def optimize(self):
        best_solution, logbook = self.run_ga()
        L_opt, d_opt = best_solution
        print("\nOptimal Parameters Found:")
        print(f"L (length): {L_opt:.4f} m")
        print(f"d (spacing): {d_opt:.4f} m")
        return L_opt, d_opt