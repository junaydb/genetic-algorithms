from SSGA import Individual, SSPopulation
import random


target = """
00011110000011000
00100001000101100
01000000001000110
00100111001111110
00011101001000010
""".strip().replace(
    "\n", ""
)


def main():
    # Hyperparameters are set via the constructor
    population = SSPopulation(
        ASCIIArtIndividual,
        size=250,
        max_gens=3000,
        crossover_chance=1,
        mutation_chance=0.01,
        selection_strat="tournament",
        win_chance=1,
        pool_size=2,
        print_results=True,
        check_for_target=True,
        print_progress=False,
        verbose=False,
        show_plot=False,
        stop_when_found=True,
    )

    ### Hyperparameters can also be set outside the constructor:
    #  population.win_chance = 0.9
    #  population.mutation_chance = 0.05
    #  population.crossover_chance = 0.5

    ### Updating `size` will re-initialise the population internally:
    #  population.size = 500

    # Runs the GA
    population.simulate()

    # For hist.py
    if population.solution_index is not None:
        return round(population.solution_index, -3) // 1000
    else:
        return -1


# Implementing the base `Individual` abstract class to create a concrete
# implementation for the ASCII art problem. Plugs directly into the `SSPopulation`
# class constructor.
class ASCIIArtIndividual(Individual):
    def get_phenotype(self):
        # The phenotype and genotype representations are the same for this
        # individual
        return self.genotype

    def crossover(self, mate):
        # Uniform crossover
        offspring_genotype = "".join(
            [
                self.genotype[i] if random.random() < 0.5 else mate.genotype[i]
                for i in range(len(self.genotype))
            ]
        )
        return ASCIIArtIndividual(offspring_genotype)

    def fitness(self):
        # Return the sum of the number of matching bits
        return sum(1 for i, j in zip(self.get_phenotype(), target) if i == j)

    @staticmethod
    def create_random():
        return ASCIIArtIndividual(
            "".join(random.choice("01") for _ in range(len(target)))
        )

    def is_target(self):
        return self.get_phenotype() == target

    def __str__(self):
        # Print the ascii art properly formatted and with colour
        string = ""
        for i, bit in enumerate(self.genotype):
            if (i + 1) % 17 == 0 and i + 1 != len(target):
                if bit == "1":
                    string += "\u001b[31m" + bit + "\u001b[0m\n"
                else:
                    string += bit + "\n"
            elif bit == "1":
                string += "\u001b[31m" + bit + "\u001b[0m"
            else:
                string += bit
        return string


if __name__ == "__main__":
    main()
