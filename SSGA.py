import numpy
import random
import more_itertools
import time
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

"""
This module contains two classes:
    - An abstract type `Individual` for representing an individual.
    - A corresponding class `SSPopulation` for holding and evolving a population 
      of individuals. Individuals in the population are expected to extend 
      `Individual`.

With this module, users can create their own specialised implementation of `Individual`
specific to their GA problem, then 'plug' their implementation into the `SSPopulation`
constructor to quickly spin-up a steady-state population with individuals of that 
type, ready to be simulated.
"""


class Individual(ABC):
    """
    Abstract type for representing and mutating an individual.

    Contains functionality for:
        - Retrieving the genotype and phenotype of this individual
        - Performing genetic operations on this individual

    Default implementations for methods are provided where it makes sense and
    can be overidden.

    Constructor:
        Creates a new individual with the passed in genotype.

    Args:
        genotype (any): genotype of this individual
    """

    def __init__(self, genotype):
        self.genotype = genotype

    @abstractmethod
    def get_phenotype(self):
        """
        Return the phenotype of this individual.

        Returns:
            any: phenotype of this individual.
        """
        pass

    @abstractmethod
    def crossover(self, mate):
        """
        Return the offspring produced from crossing over this individual with
        the passed in individual.

        Args:
            mate (same type as caller): individual to crossover with

        Returns:
            same type as caller: resulting offspring
        """
        pass

    @abstractmethod
    def fitness(self):
        """
        Return the fitness of this individual.

        Returns:
            float: fitness value
        """
        pass

    @abstractmethod
    def create_random():
        """
        Return a new individual with a random genotype.

        Returns:
            same type as caller: a new, random individual
        """
        pass

    def mutate(self):
        """
        Mutate this individual's genotype.

        Warn:
            This method operates in-place (returns `None`).
        """
        # Toss a coin and if it returns true, add a random index to bitflip_indexes.
        # Flip all the bits at the indexes contained in bitflip_indexes.
        # The chance of the coin returning true diminishes on each flip.
        mutation_mult = 1
        bitflip_indexes = []
        while random.random() < mutation_mult:
            mutation_mult *= 0.9
            bitflip_indexes.append(random.choice(range(len(self.genotype))))

        flip = lambda x: "1" if x == "0" else "0"
        self.genotype = "".join(
            flip(bit) if i in bitflip_indexes else bit
            for i, bit in enumerate(self.genotype)
        )

    def clone(self):
        """
        Return an individual with the same genotype as this individual.

        Returns:
            same type as caller: cloned individual
        """
        return self.__class__(self.genotype)

    def is_target(self):
        """
        Return a bool indicating whether this individual is the target solution.

        Returns:
            bool: True if this individual is the target, False otherwise
        """
        raise NotImplementedError(
            "is_target() must be implemented on individuals if check_for_target is enabled"
        )

    def __repr__(self):
        return f"{self.get_phenotype()}"

    def __str__(self):
        return f"{self.get_phenotype()}"

    def _cointoss(self, bias):
        return random.random() < bias


class SSPopulation:
    """
    Class for holding and mutating a steady-state population.
    Contains methods for:
        - Setting hyperparameters
        - Querying the state of the population
        - Running the steady-state GA

    Constructor:
        Creates a new population containing `size` individuals of type
        `individual_t`, sets hyperparameters, and sets options for statistics
        reporting behaviour.

    Warn:
        Enabling printing and/or plotting has a large impact on performance.

    Args:
        individual_t (subclass of `Individual`): type of individuals in this population
        size (int): number of individuals the generated population will have
        max_gens (int): maximum number of iterations
        crossover_chance (float): probability of producing an 'offspring' from two
                                  current individuals, implicitly sets the chance
                                  of cloning to 1 - crossover_chance
        mutation_chance (float): probability of an individual being mutated
        selection_stat ("tournament" | "fps"): selection strategy to use
        win_chance (float): tournament win probability
        pool_size (int): number of individuals to sample for the tournament
        check_for_target (bool): enable checking for a target solution, requires
                                 individual_t to have is_target() implemented
        stop_when_found (bool): stop the simulation when the target has been found
        print_results (bool): print information about the result of the GA
        print_progress (bool): print current generation index and highest fitness
                         each iteration
        verbose (bool): print information about the fittest individual each
                        iteration and about mutations when they occur
        show_plot (bool): display a plot showing max fitness vs average fitness
                          over all generations
    """

    def __init__(
        self,
        individual_t,
        size=100,
        max_gens=1000,
        crossover_chance=0.75,
        mutation_chance=0.001,
        selection_strat="tournament",
        win_chance=1,
        pool_size=3,
        check_for_target=False,
        stop_when_found=False,
        print_results=False,
        print_progress=False,
        verbose=False,
        show_plot=False,
    ):
        if not issubclass(individual_t, Individual):
            raise TypeError("individual_t needs to be a subclass of `Individual`")

        self.__size = size
        self.__individual_t = individual_t
        self.__population = [self.__individual_t.create_random() for _ in range(size)]
        self.__solution_index = None
        self.__gen_index = 0

        # Hyperparameters
        self.max_gens = max_gens
        self.crossover_chance = crossover_chance
        self.mutation_chance = mutation_chance
        self.select_strat = selection_strat
        self.win_chance = win_chance
        self.pool_size = pool_size

        # Stuff that determines what is reported back to the user
        self.check_for_target = check_for_target
        self.stop_when_found = stop_when_found
        self.print_results = print_results
        self.print_progress = print_progress
        self.verbose = verbose
        self.show_plot = show_plot
        if show_plot:
            self.__max_fitnesses = []
            self.__avg_fitnesses = []

    def simulate(self):
        """
        Run the steady-state GA.
        """
        start = time.time()

        while self.__gen_index < self.max_gens:
            if self.show_plot:
                self.__track_fitness()

            ### SELECTION ###
            if self.select_strat == "tournament":
                if self.__cointoss(self.crossover_chance):
                    parent_a = self.__tournament_selection()
                    parent_b = self.__tournament_selection()
                    offspring = parent_a.crossover(parent_b)
                else:
                    parent = self.__tournament_selection()
                    offspring = parent.clone()
                elder = self.__tournament_selection(high_fitness=False)
            elif self.select_strat == "fps":
                if self.__cointoss(self.crossover_chance):
                    parent_a = self.__FPS()
                    parent_b = self.__FPS()
                    offspring = parent_a.crossover(parent_b)
                else:
                    parent = self.__FPS()
                    offspring = parent.clone()
                elder = self.__FPS(high_fitness=False)
            else:
                raise Exception(
                    f"{self.select_strat} is not a valid selection strategy"
                )

            ### MUTATION ###
            if self.__cointoss(self.mutation_chance):
                if self.print_progress:
                    print("MUTATING")

                if self.print_progress and self.verbose:
                    print("Before mutation:")
                    print(f"Fitness: {offspring.fitness()}")
                    print(offspring)

                offspring.mutate()

                if self.print_progress and self.verbose:
                    print("After mutation:")
                    print(f"Fitness: {offspring.fitness()}")
                    print(offspring)

            # Replace low fitness individual with offspring
            elder.genotype = offspring.genotype

            if self.print_progress:
                sorted_population = sorted(
                    self.__population,
                    key=lambda individual: individual.fitness(),
                    reverse=True,
                )
                print(f"Gen index: {self.__gen_index + 1}")
                print(f"Max fitness: {sorted_population[0].fitness()}")
                print(sorted_population[0])

            # If a target solution was provided, check if it's been discovered,
            # and stop the simulation if stop_when_found is enabled.
            if self.check_for_target is True:
                self.__check_for_target()
                if self.solution_index is not None and self.stop_when_found:
                    self.__gen_index += 1
                    break

            self.__gen_index += 1
            # End of simulation loop

        # Store the time taken in ms
        end = time.time()
        self.sim_time = (end - start) * 1000

        # Report results of the simulation
        if self.print_results:
            self.__display_sim_stats()
        if self.show_plot:
            self.__display_plot()

        # Return the best solution if no target solution was provided
        if self.check_for_target is None:
            return self.population[0]

    def reset(self):
        """
        Reset state and re-initialise the population.
        """
        self.__gen_index = 0
        self.__population = [
            self.__individual_t.create_random() for _ in range(self.size)
        ]
        self.__max_fitnesses = []
        self.__avg_fitnesses = []

    def __tournament_selection(self, high_fitness=True):
        """
        Run a tournament containing `size` randomly selected
        individuals and return the winner.

        Args:
            size (int): number of individuals in the tournament
            high_fitness (bool): if false, low fitness individuals have a higher
                                 chance of winning

        Returns:
            Individual: tournament winner
        """
        pool = random.sample(self.__population, self.pool_size)
        pool.sort(key=lambda individual: individual.fitness(), reverse=high_fitness)

        for individual in pool:
            if self.__cointoss(self.win_chance):
                return individual
        else:
            # Return the last individual in the pool if no one wins
            return pool[-1]

    def __FPS(self, high_fitness=True):
        """
        Return an individual that has been selected from the population via
        fitness proportion selection.

        Args:
            high_fitness (bool): if false, low fitness individuals have larger
                                 proportions

        Returns:
            Individual: individual selected via fitness proportion selection
        """
        fitness_scores = [individual.fitness() for individual in self.__population]
        min_fitness = min(fitness_scores)
        # No negative fitness scores
        fitness_scores = [
            (0.1 * min_fitness) + score - min_fitness for score in fitness_scores
        ]

        if high_fitness == False:
            # Give low fitness individuals a higher probability of being selected
            max_fitness = max(fitness_scores)
            fitness_scores = [
                (0.1 * max_fitness) + max_fitness - score for score in fitness_scores
            ]

        r = sum(fitness_scores) * random.random()
        lower_bound = [sum(fitness_scores[:i]) for i in range(self.size)]
        upper_bound = [sum(fitness_scores[: i + 1]) for i in range(self.size)]
        selected_index = more_itertools.first_true(
            zip(range(self.size), lower_bound, upper_bound),
            pred=lambda T: T[1] <= r < T[2],
        )[0]
        return self.__population[selected_index]

    def __sort_by_fitness(self, high_fitness=True):
        """
        Sort the population by fitness in-place.

        Warn:
            This method sorts in-place (returns `None`).

        Args:
            high_fitness (bool): if true, sort in descending fitness order,
                                 otherwise sort in ascending fitness order
        """
        self.__population.sort(
            key=lambda individual: individual.fitness(), reverse=high_fitness
        )

    def __cointoss(self, bias):
        return random.random() < bias

    def __check_for_target(self):
        """
        Check if the target solution has been found.
        """
        # If the target solution has been found, store the gen index it was found at.
        if self.__solution_index is None and any(
            i.is_target() for i in self.__population
        ):
            self.__solution_index = self.__gen_index

    def __display_sim_stats(self):
        """
        Print results of the simulation.
        """
        self.__sort_by_fitness()

        if self.check_for_target is False:
            print(f"MAX GENERATION LIMIT ({self.max_gens}) REACHED")
            print(f"Highest fitness: {self.__population[0].fitness()}")
            print(self.__population[0])
            print(f"Time taken: {round(self.sim_time, 2)}ms")
        elif self.__gen_index != self.max_gens:
            print("SOLUTION FOUND, STOPPING")
            print(f"Solution found in generation {self.__solution_index}")
            print(f"Fitness: {self.__population[0].fitness()}")
            print(self.__population[0])
            print(f"Time taken: {round(self.sim_time, 2)}ms")
        else:
            print(f"MAX GENERATION LIMIT ({self.max_gens}) REACHED")
            if self.__solution_index is None:
                print(f"Solution not found by generation {self.__gen_index}")
                print(f"Highest fitness: {self.__population[0].fitness()}")
                print(self.__population[0])
                print(f"Time taken: {round(self.sim_time, 2)}ms")
            else:
                print(f"Solution found in generation {self.__solution_index}")
                print(f"Fitness: {self.__population[0].fitness()}")
                print(self.__population[0])
                print(f"Time taken: {round(self.sim_time, 2)}ms")

    def __track_fitness(self):
        """
        Get the max and mean fitness for the current population and append the
        them to the corresponding member lists.

        Warn:
            Significantly increases runtime
        """
        fitnesses = numpy.array([i.fitness() for i in self.__population])
        self.__max_fitnesses.append(fitnesses.max())
        self.__avg_fitnesses.append(fitnesses.mean())

    def __display_plot(self):
        """
        Plot the max and average fitness for each generation and display it.
        """
        x = range(self.__gen_index)
        plt.plot(x, self.__max_fitnesses, label="Max")
        plt.plot(x, self.__avg_fitnesses, label="Avg")
        plt.legend()
        plt.title("Avg vs Max Fitness per Generation")
        plt.show(block=True)

    @property
    def size(self):
        return self.__size

    @size.setter
    def size(self, size):
        """
        `size` setter. Re-initialises the population with the new size.

        Args:
            size (int): new size
        """
        self.__size = size
        self.__population = [self.__individual_t.create_random() for _ in range(size)]

    # Attempting to set these externally will throw
    @property
    def population(self):
        return self.__population

    @property
    def solution_index(self):
        return self.__solution_index

    @property
    def individual_t(self):
        return self.__individual_t
