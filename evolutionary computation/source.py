# Run with python 3

import os.path
import sys
import random
import math

## VARS ##
GRID_SIZE = 100  # grid of size nXn

# SRC_POINT = (10, 10)  # Maze 1 coordinates
# DEST_POINT = (50, 50)  # Maze 1  coordinates

SRC_POINT = (0, 0)  # Maze 2 coordinates
DEST_POINT = (50, 50)  # Maze 2 coordinates

MAZE_FILE = "mmaze.bmp"  # only 1 bbp bmp are supported

# GA VARS
GENERATIONS = 5000
GENERATIONS_STALEMATE = 500  # Stop after GENERATIOIN_STALEMATE generations without improvement
DNA_SIZE = int(GRID_SIZE * 2.75)
POPULATION_SIZE = 60
MUTATION_CHANCE = 0.09
CROSSOVER_CHANCE = 0.95
ELITISEM_PRECENTAGE = 0.05

# DO NOT TOUCH
OBSTACLES_MAP = [[False for x in range(GRID_SIZE)] for y in range(GRID_SIZE)]
MAZE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), MAZE_FILE)
SOLVED = False

TEST_NAME = "size{}pop{}maze{}".format(GRID_SIZE, POPULATION_SIZE, MAZE_FILE)
STATS_GEN_FITNESS_MAXES = []
STATS_GEN_FITNESS_AVGS = []
STATS_GEN_FITNESS_MINS = []


## FUNCTIONS ##
def manhattan_distance(p1, p2):
    "Returns the Manhattan distance between points p1 and p2"
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def maze_walk(directions, source=SRC_POINT, destination=DEST_POINT):
    """
    Walks the maze by the provided directions, and gathers information about the route.
    Returns a tuple of (distance, repeated_nodes, obstacles, final_point, is_out_of_bound).

    Distance will be infinity if the directions are invalid (are taking us out of scope, or not reaching
    the destination)
    """
    if source == destination:
        return 0, 0, 0, source, False

    p = source
    distance = 0
    repeated_nodes = 0
    obstacles = 0
    points_visited = [p]
    for direction in directions:
        distance += 1

        if direction == "L":
            p = (p[0] - 1, p[1])
        elif direction == "R":
            p = (p[0] + 1, p[1])
        elif direction == "U":
            p = (p[0], p[1] - 1)
        elif direction == "D":
            p = (p[0], p[1] + 1)
        else:
            raise ValueError("Direction is invalid.")

        if p[0] < 0 or p[0] >= GRID_SIZE or p[1] < 0 or p[1] >= GRID_SIZE:
            # out of bounds, illegal move
            return float('inf'), repeated_nodes, obstacles, p, True

        if OBSTACLES_MAP[p[0]][p[1]]:
            obstacles += 1

        if p in points_visited:
            repeated_nodes += 1
        else:
            points_visited.append(p)

        if p == destination:
            return distance, repeated_nodes, obstacles, p, False

    return float('inf'), repeated_nodes, obstacles, p, False


def weighted_choice(items):
    """
    Chooses a random element from items, where items is a list of tuples in
    the form (item, weight). weight determines the probability of choosing its
    respective item.
    """
    weight_total = sum((item[1] for item in items))
    n = random.uniform(0, weight_total)
    for item, weight in items:
        if n < weight:
            return item
        n -= weight
    return item


def random_gene(exclude=None):
    "Chooses a random gene out of the directions U(p), D(own), L(eft), R(ight)."

    choices = ["U", "D", "L", "R"]

    if exclude:
        choices.remove(exclude)

    return random.choice(choices)


def random_chromosome():
    "Creates a random chromosome of size DNA_SIZE."
    return "".join([random_gene() for _ in range(DNA_SIZE)])


def random_population():
    "Creates a random population of size POPULATION_SIZE."
    return [random_chromosome() for _ in range(POPULATION_SIZE)]


def read_obstacles_map():
    mat = read_1bpp_bmp(MAZE_PATH)
    if len(mat) != GRID_SIZE or len(mat[0]) != GRID_SIZE:
        raise ValueError(
            "Maze file size is {}x{}, but the our grid size is {}.".format(len(mat), len(mat[0]), GRID_SIZE))
    if mat[SRC_POINT[0]][SRC_POINT[1]]:
        raise ValueError("Starting point {} is in an obstacle.".format(SRC_POINT))
    if mat[DEST_POINT[0]][DEST_POINT[1]]:
        raise ValueError("Destination point {} is in an obstacle.".format(DEST_POINT))
    return mat


def random_obstacles_map(OBSTACLE_RATIO):
    """
    Creates a random obstacles map with OBSTACLE_RATIO % of the map being obstacles.
    Does not create an obstacle at SRC_POINT or DEST_POINT.
    Returns a matrix with boolean values which indicates whether the cell contains an obstacle or not.
    """
    obstacles_left = int(round(OBSTACLE_RATIO * GRID_SIZE ** 2))
    mat = [[False for x in range(GRID_SIZE)] for y in range(GRID_SIZE)]

    while obstacles_left > 0:
        # creating random x,y
        x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)

        # if the cell is not a source point, dest point, or already has an obstacle, place an obstacle in it
        if not mat[x][y] and (x, y) != SRC_POINT and (x, y) != DEST_POINT:
            mat[x][y] = True
            obstacles_left -= 1

    return mat


def fitness(dna):
    "Computes the fitness of a specific chromosome."
    global SOLVED

    REPEATED_PENALTY = 2  # each repeated node is considered the same as REPEATED_PENALTY steps
    OBSTACLE_PENALTY = 50  # each obstacle on the way is considered the same as OBSTACLE_PENALTY steps
    DEST_NOT_REACHED_PENALTY = manhattan_distance(SRC_POINT, DEST_POINT) + DNA_SIZE  # if destination is not reached

    distance, repeated_nodes, obstacles, final_point, is_out_of_bound = maze_walk(dna)

    # Calculate Penalties
    penalties = repeated_nodes * REPEATED_PENALTY + obstacles * OBSTACLE_PENALTY
    if is_out_of_bound:
        penalties = float('inf')
    elif distance == float('inf'):
        penalties += DEST_NOT_REACHED_PENALTY

    if distance == float('inf'):
        weighted_cost = manhattan_distance(final_point, DEST_POINT) + penalties
    else:
        weighted_cost = distance + penalties

    if distance != float('inf') and not obstacles:
        SOLVED = True

    # returning the inverse of the cost, so it'll become a proper fitness function
    return 1.0 / weighted_cost


def mutate(dna):
    """
    The mutation function. For each gene in the DNA, there's a MUTATION_CHANCE
    that it'll get replaced by a random gene.
    """
    output = ""
    for c in dna:
        if random.random() < MUTATION_CHANCE:
            output += random_gene(exclude=c)
        else:
            output += c
    return output


def crossover_single_point(dna1, dna2):
    "Single-point crossover"
    pos = int(random.random() * DNA_SIZE)
    return (dna1[:pos] + dna2[pos:], dna2[:pos] + dna1[pos:])


def crossover_k_point(dna1, dna2, k):
    "K-point crossover"

    for _ in range(k):
        dna1, dna2 = crossover_single_point(dna1, dna2)

    return dna1, dna2


def crossover_two_point(dna1, dna2):
    "Two-point crossover"
    return crossover_k_point(dna1, dna2, k=2)


def crossover_uniform(dna1, dna2, p=0.5):
    "Uniform crossover"

    output_dna1 = ""
    output_dna2 = ""
    for i in range(DNA_SIZE):
        if random.random() < p:
            output_dna1 += dna1[i]
            output_dna2 += dna2[i]
        else:
            output_dna1 += dna2[i]
            output_dna2 += dna1[i]

    return output_dna1, output_dna2


def read_1bpp_bmp(path):
    with open(path, 'rb') as f:
        # find width
        f.seek(0x12)
        width = f.read(1)[0]
        # find height
        f.seek(0x16)
        height = f.read(1)[0]

        mat = [[False for x in range(width)] for y in range(height)]

        f.seek(0x3e)
        for row in range(height):
            # read the num. of bytes required to read the row
            chunk = f.read(int(math.ceil(math.ceil(width / 8.0) / 4) * 4))
            for col in range(width):
                mat[row][col] = (chunk[int(col / 8)] & 2 ** ((7 - col) % 8)) >> ((7 - col) % 8) == 0

    # reverse mat
    for i in range(int(len(mat) / 2)):
        j = len(mat) - i - 1
        t = mat[i]
        mat[i] = mat[j]
        mat[j] = t

    return mat


def print_1bpp_bmp_mat(mat):
    sys.stdout.write("\t  {}".format("".join([str(x + 1) for x in range(len(mat[0]))])))
    for x in range(len(mat)):
        sys.stdout.write("\n{}\t".format(x + 1))

        for y in range(len(mat[0])):
            if mat[x][y]:
                sys.stdout.write("*")
            else:
                sys.stdout.write(" ")


## MAIN ##
def main():
    global OBSTACLES_MAP, SOLVED, STATS_GEN_FITNESS_MAXES, STATS_GEN_FITNESS_AVGS, STATS_GEN_FITNESS_MINS

    # Create random population, sorted by fitness
    population = random_population()
    population.sort(key=lambda x: fitness(x), reverse=True)
    # OBSTACLES_MAP = random_obstacles_map(0.1)
    OBSTACLES_MAP = read_obstacles_map()
    SOLVED = False

    best_fitness = 0
    same_fitness_count = 0
    for generation in range(0, GENERATIONS):
        # calc statistics
        STATS_GEN_FITNESS_MAXES.append(fitness(population[0]))
        STATS_GEN_FITNESS_MINS.append(fitness(population[-1]))
        STATS_GEN_FITNESS_AVGS.append(float(sum([fitness(x) for x in population])) / len(population))

        # stop if we're in stalemate
        new_best_fitness = fitness(population[0])
        if best_fitness == new_best_fitness:
            same_fitness_count += 1
        else:
            same_fitness_count = 0

        if same_fitness_count >= GENERATIONS_STALEMATE:
            print("Stopping after {} generations without any improvement (stalemate).".format(same_fitness_count))
            break

        best_fitness = new_best_fitness
        best_fitness_inverted = 1.0 / best_fitness if best_fitness != 0 else float('inf')
        print("Gen {}...\tBest: '{}'\tFitness: (1/{:.2f})\tSolved: {}".format(generation + 1, population[0],
                                                                              best_fitness_inverted, SOLVED))

        weighted_population = [(x, fitness(x)) for x in population]

        # Move ELITISEM_PRECENTAGE of the best population to the next generation
        elitisem_amount = int(round(ELITISEM_PRECENTAGE * POPULATION_SIZE))
        population = population[:elitisem_amount]

        for _ in range(int(POPULATION_SIZE / 2)):
            # Selection
            ind1 = weighted_choice(weighted_population)
            ind2 = weighted_choice(weighted_population)

            # Crossover
            if random.random() < CROSSOVER_CHANCE:
                # ind1, ind2 = crossover_uniform(ind1, ind2)
                ind1, ind2 = crossover_single_point(ind1, ind2)
                # ind1, ind2 = crossover_two_point(ind1, ind2)

            # Mutate and add back into the population.
            population.append(mutate(ind1))
            population.append(mutate(ind2))

        # sort by fitness
        population.sort(key=lambda x: fitness(x), reverse=True)

        # remove worst invidiauls (with smallest fitness) from the
        # popluation, and trim it back to POPULATION_SIZE.
        population = population[:POPULATION_SIZE]

    # Print GA performance

    distance, repeated_nodes, obstacles, final_point, is_out_of_bound = maze_walk(population[0])
    best_fitness = fitness(population[0])
    best_fitness_inverted = 1.0 / best_fitness if best_fitness != 0 else float('inf')
    stabilized_generation = generation + 1 - same_fitness_count

    print("*** DONE ***\r\n\r\n")
    print("PATH FOUND:\t\t\t{}\r\n\r\n".format("YES" if distance != float('inf') else "NO"))
    print("Generation: {} (Stabilized at generation {})".format(generation + 1, stabilized_generation + 1))
    print("Best chromosome: '{}'".format(population[0]))
    print("Fitness: {:.2f} (1/{:.2f})".format(best_fitness, best_fitness_inverted))
    print("Distance from destination: {}".format(distance))
    print("repeated_nodes: {}".format(repeated_nodes))
    print("obstacles: {}".format(obstacles))
    print("final_point: {}".format(final_point))
    print("Population Count: {} ({} unique)".format(len(population), len(set(population))))

    return distance != float('inf'), stabilized_generation


if __name__ == "__main__":
    main()
    with open('{}_max.txt'.format(TEST_NAME), 'w') as f:
        for n in STATS_GEN_FITNESS_MAXES: f.write("{}\n".format(n))
    with open('{}_avg.txt'.format(TEST_NAME), 'w') as f:
        for n in STATS_GEN_FITNESS_AVGS: f.write("{}\n".format(n))
    with open('{}_min.txt'.format(TEST_NAME), 'w') as f:
        for n in STATS_GEN_FITNESS_MINS: f.write("{}\n".format(n))