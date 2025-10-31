from SSGA import Individual, SSPopulation
import random


def main():
    # Hyperparameters for single-objective optimisation function
    population = SSPopulation(
        RealEncodedIndividual,
        size=1000,
        max_gens=10000,
        crossover_chance=0.8,
        mutation_chance=0.04,
        selection_strat="tournament",
        win_chance=1,
        pool_size=2,
        print_results=True,
        print_progress=False,
        show_plot=False,
    )

    for i in range(10):
        population.reset()
        population.simulate()
        print()

    # Hyperparameters for constrained optimisation function
    population = SSPopulation(
        RealEncodedIndividual,
        size=2000,
        max_gens=10000,
        crossover_chance=0.9,
        mutation_chance=0.06,
        selection_strat="tournament",
        win_chance=1,
        pool_size=2,
        print_results=True,
        print_progress=False,
    )

    #  for i in range(10):
    #  population.reset()
    #  population.simulate()
    #  highest_fitness = population.population[0].fitness()
    #  print(format(highest_fitness, ".8f"))
    #  print()


class RealEncodedIndividual(Individual):
    # Decode to real-value pair
    def get_phenotype(self):
        n_bits = len(self.genotype) // 2

        min_val_xy, max_val_xy = active_search_domain

        resolution_xy = (max_val_xy - min_val_xy) / (2**n_bits - 1)

        x_genotype, y_genotype = self.genotype[:n_bits], self.genotype[n_bits:]

        x_phenotype = min_val_xy + resolution_xy * int(x_genotype, 2)
        y_phenotype = min_val_xy + resolution_xy * int(y_genotype, 2)

        return x_phenotype, y_phenotype

    def crossover(self, mate):
        # Uniform crossover
        offspring_genotype = "".join(
            [
                self.genotype[i] if random.random() < 0.5 else mate.genotype[i]
                for i in range(len(self.genotype))
            ]
        )
        return RealEncodedIndividual(offspring_genotype)

    def fitness(self):
        x, y = self.get_phenotype()
        return -active_fitness_function(x, y)

    @staticmethod
    def create_random():
        return RealEncodedIndividual(
            "".join(random.choice("01") for _ in range(n_bits))
        )


booth_search_domain = [-10.0, 10.0]
rosenbrock_search_domain = [-1.5, 1.5]


def booth_function(x, y):
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2


# Rosenbrock function constrained to a disk.
# I.e. Rosenbrock function subjected to: x^2 + y^2 <= 2
def rosenbrock_constrained_to_disk(x, y):
    rosenbrock_function = lambda x, y: (1 - x) ** 2 + 100 * (y - x**2) ** 2
    constraint = lambda x, y: x**2 + y**2
    penalty_coeff = 100
    penalty = lambda constraint: (constraint - 2) ** 2 if constraint > 2 else 0
    return rosenbrock_function(x, y) + penalty_coeff * penalty(constraint(x, y))


# Set fitness function to use
active_fitness_function = booth_function

# Derive the search domain and encoding resolution based on the selected function
if active_fitness_function == booth_function:
    active_search_domain = booth_search_domain
    n_bits = 22
else:
    active_search_domain = rosenbrock_search_domain
    n_bits = 16


if __name__ == "__main__":
    main()
