from SSGA import Individual, SSPopulation
from itertools import permutations
import random


letters = ["A", "C", "L", "M", "N", "S", "T", "U", "X"]


def main():
    # Hyperparameters are set via the constructor
    population = SSPopulation(
        SantaClausRiddleIndividual,
        size=500,
        max_gens=5000,
        crossover_chance=0.9,
        mutation_chance=0.001,
        selection_strat="tournament",
        win_chance=1,
        pool_size=2,
        print_results=True,
        check_for_target=False,
        print_progress=False,
        verbose=False,
        show_plot=False,
        stop_when_found=True,
    )

    #  population.simulate()
    for i in range(10):
        population.reset()
        population.simulate()

    #  For hist.py
    if population.solution_index is not None:
        return round(population.solution_index, -3) // 1000
    else:
        return -1


class SantaClausRiddleIndividual(Individual):
    def get_phenotype(self):
        map = dict(zip(letters, self.genotype))
        self.map = map  # For use in __str__()

        santa = int(f"{map['S']}{map['A']}{map['N']}{map['T']}{map['A']}")
        claus = int(f"{map['C']}{map['L']}{map['A']}{map['U']}{map['S']}")
        xmas = int(f"{map['X']}{map['M']}{map['A']}{map['S']}")

        return santa, claus, xmas

    def crossover(self, mate):
        unique = set()
        while len(unique) < len(self.genotype):
            # Uniform crossover
            offspring_genotype = [
                self.genotype[i] if random.random() < 0.5 else mate.genotype[i]
                for i in range(len(self.genotype))
            ]
            unique = set(offspring_genotype)
        return SantaClausRiddleIndividual(offspring_genotype)

    def mutate(self):
        a, b = random.sample(range(len(self.genotype)), 2)
        self.genotype[a], self.genotype[b] = self.genotype[b], self.genotype[a]

    def fitness(self):
        santa, claus, xmas = self.get_phenotype()
        return -abs((santa - claus) - xmas)

    @staticmethod
    def create_random():
        genotype = [digit for digit in range(10)]
        random.shuffle(genotype)
        return SantaClausRiddleIndividual(genotype[:9])

    # Helper for brute-forcing
    @staticmethod
    def create_with_genotype(genotype):
        return SantaClausRiddleIndividual(genotype)

    def is_target(self):
        return self.fitness() == 0

    # Show the mapping in the string output
    def __str__(self):
        return f"{self.get_phenotype()}\n{self.map}"


# To check what all the possible solutions are
def brute_force_search():
    num_solutions = 0
    perms = list(permutations(range(10)))
    for perm in perms:
        individual = SantaClausRiddleIndividual.create_with_genotype(perm)
        if individual.fitness() == 0:
            num_solutions += 1
            print(individual.get_phenotype())
    print(f"TOTAL SOLUTIONS: {num_solutions}")


if __name__ == "__main__":
    main()
