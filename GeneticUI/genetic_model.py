import random
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import svm

# dataset = load_boston()
# X, y = dataset.data, dataset.target
# features = dataset.feature_names
# estimator = LinearRegression()
# print(np.mean(-1.0*(cross_val_score(estimator, X, y, scoring='neg_mean_squared_error'))))

class GeneticAlgorithm():
    def __init__(self, estimator,
                 num_of_generations,
                 population_size,
                 mutation_rate,
                 num_best_to_select,
                 num_random_to_select,
                 num_children):
        self.estimator = estimator
        self.num_of_generations = num_of_generations
        self.population_size = population_size
        self.num_best_to_select = num_best_to_select
        self.num_random_to_select = num_random_to_select
        self.num_children = num_children
        self.mutation_rate = mutation_rate

    def initialize(self):
        population = []
        # generate an initial population in which each chromosome is a vector of same size
        # as number of features and the chromosome is a boolean vector which essentially means
        # a set of True or False values which determine which feature gets dropped in a iteration.
        for i in range(self.population_size):
            chromosome = np.ones(self.num_of_features, dtype=np.bool)
            mask = np.random.rand(len(chromosome)) < 0.3
            chromosome[mask] = False
            population.append(chromosome)
        return population

    def get_r2_score(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
        self.estimator.fit(X_train, y_train)
        y_pred = self.estimator.predict(X_test)
        score = r2_score(y_test, y_pred)
        return score*100

    def fitness(self, population):
        X, y = self.dataset
        scores = []
        for chromosome in population:
            score = -1.0 * np.mean(cross_val_score(self.estimator, X[:,chromosome], y, cv=5, scoring='neg_mean_squared_error'))
            scores.append(score)
        scores, population = np.array(scores), np.array(population)
        inds = np.argsort(scores)
        return list(scores[inds]), list(population[inds, :])

    def select(self, population_sorted):
        population_next = []
        for i in range(self.num_best_to_select):
            population_next.append(population_sorted[i])
        for i in range(self.num_random_to_select):
            population_next.append(random.choice(population_sorted))
        random.shuffle(population_next)
        return population_next

    def crossover(self, population):
        population_next = []
        for i in range(int(len(population)/2)):
            for j in range(self.num_children):
                chromosome1, chromosome2 = population[i], population[len(
                    population)-1-i]
                child = chromosome1
                mask = np.random.rand(len(child)) > 0.5
                child[mask] = chromosome2[mask]
                population_next.append(child)
        return population_next

    def mutate(self, population):
        population_next = []
        for i in range(len(population)):
            chromosome = population[i]
            if random.random() < self.mutation_rate:
                mask = np.random.rand(len(chromosome)) < self.mutation_rate
                chromosome[mask] = False
            population_next.append(chromosome)
        return population_next

    def generate(self, population):
        # Selection, crossover and mutation
        scores_sorted, population_sorted = self.fitness(population)
        population = self.select(population_sorted)
        population = self.crossover(population)
        population = self.mutate(population)
        # History
        self.chromosomes_best.append(population_sorted[0])
        self.scores_best.append(scores_sorted[0])
        self.scores_avg.append(np.mean(scores_sorted))
        return population

    def get_best_features(self, features):
        feature_list = []
        if(self.chromosomes_best):
            for chromosome in self.chromosomes_best:
                feature_list.append(features[chromosome])
        return feature_list
    def fit(self, X, y):
        self.chromosomes_best = []
        self.scores_best, self.scores_avg = [], []
        self.dataset = (X, y)
        self.num_of_features = X.shape[1]
        population = self.initialize()
        for i in range(self.num_of_generations):
            population = self.generate(population)
        return self


# GA = GeneticAlgorithm(estimator=LinearRegression(),
#                       num_of_generations=7, population_size=200, num_best_to_select=40, num_random_to_select=40,
#                       num_children=5, mutation_rate=0.05)

# GA.fit(X, y)
# final_score = np.mean(-1.0*cross_val_score(estimator, X[:,GA.chromosomes_best[-1]], y, scoring='neg_mean_squared_error'))
# print(features[GA.chromosomes_best[-1]])
# print(final_score)