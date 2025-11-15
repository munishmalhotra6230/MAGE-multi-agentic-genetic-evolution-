import numpy as np
import random

# ---------- Hyperparameters ----------
POP_SIZE = 50
GENS = 100
DIM = 1
BOUNDS = [(-1.0, 2.0)]
TOURNAMENT_SIZE = 3
CROSSOVER_PROB = 0.9
MUTATION_PROB = 0.3
MUTATION_SCALE = 0.05
ELITISM = 1
SEED = 1
np.random.seed(SEED)
random.seed(SEED)

# ---------- Fitness ----------
def fitness_function(x):
    return x * np.sin(10 * np.pi * x) + 1.0

# ---------- Population helpers ----------
def create_population(pop_size, dim, bounds):
    pop = np.zeros((pop_size, dim))
    for d in range(dim):
        lo, hi = bounds[d]
        pop[:, d] = np.random.uniform(lo, hi, size=pop_size)
    return pop

def evaluate_population(pop):
    if pop.shape[1] == 1:
        return fitness_function(pop[:, 0])
    else:
        return np.array([fitness_function(ind) for ind in pop])

# ---------- Selection / Crossover / Mutation ----------
def tournament_selection(pop, fitness, k):
    idx = np.random.choice(len(pop), size=k, replace=False)
    best = idx[np.argmax(fitness[idx])]
    return pop[best].copy()

def arithmetic_crossover(p1, p2):
    alpha = np.random.uniform(size=p1.shape)
    c1 = alpha * p1 + (1 - alpha) * p2
    c2 = alpha * p2 + (1 - alpha) * p1
    return c1, c2

def mutate(ind, bounds, mutation_scale):
    out = ind.copy()
    for i in range(out.shape[0]):
        lo, hi = bounds[i]
        scale = (hi - lo) * mutation_scale
        out[i] += np.random.normal(0, scale)
        out[i] = np.clip(out[i], lo, hi)
    return out

# ---------- Next generation ----------
def next_generation(pop, fitness):
    elite_idx = np.argsort(fitness)[-ELITISM:]
    elites = pop[elite_idx].copy()
    new_pop = []

    while len(new_pop) < POP_SIZE - ELITISM:
        p1 = tournament_selection(pop, fitness, TOURNAMENT_SIZE)
        p2 = tournament_selection(pop, fitness, TOURNAMENT_SIZE)
        if np.random.rand() < CROSSOVER_PROB:
            c1, c2 = arithmetic_crossover(p1, p2)
        else:
            c1, c2 = p1.copy(), p2.copy()
        if np.random.rand() < MUTATION_PROB:
            c1 = mutate(c1, BOUNDS, MUTATION_SCALE)
        if np.random.rand() < MUTATION_PROB:
            c2 = mutate(c2, BOUNDS, MUTATION_SCALE)
        new_pop.append(c1)
        if len(new_pop) < POP_SIZE - ELITISM:
            new_pop.append(c2)

    new_pop = np.vstack(new_pop + list(elites))
    return new_pop

# ---------- Main GA ----------
def run_ga():
    pop = create_population(POP_SIZE, DIM, BOUNDS)
    fitness = evaluate_population(pop)

    best_history = []
    mean_history = []
    std_history = []

    for gen in range(1, GENS + 1):
        pop = next_generation(pop, fitness)
        fitness = evaluate_population(pop)

        best = np.max(fitness)
        mean = np.mean(fitness)
        std = np.std(fitness)
        best_history.append(best)
        mean_history.append(mean)
        std_history.append(std)

        if gen % 10 == 0 or gen == 1 or gen == GENS:
            print(f"Gen {gen:3d} | best {best:.6f} | mean {mean:.6f} | std {std:.6f}")

    best_idx = np.argmax(fitness)
    return {
        "best_individual": pop[best_idx],
        "best_fitness": fitness[best_idx],
        "history": {"best": np.array(best_history), "mean": np.array(mean_history), "std": np.array(std_history)}
    }

if __name__ == "__main__":
    res = run_ga()
    print("\nBest found solution:")
    print("x =", res["best_individual"])
    print("fitness =", res["best_fitness"])

        
        
           




