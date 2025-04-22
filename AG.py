import numpy as np
import matplotlib.pyplot as plt
import random
import time

class GeneticAlgorithm:
    def __init__(self, dist_matrix, population_size=100, elite_size=20, mutation_rate=0.01, generations=500):
        self.dist_matrix = dist_matrix
        self.num_cities = len(dist_matrix)
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        
        self.best_distance_history = []
        self.avg_distance_history = []
        
        self.population = self.initial_population()
        
        self.ranked_population = self.rank_routes(self.population)
        self.initial_distance = 1 / self.ranked_population[0][1]
        
        self.best_route = None
        self.best_distance = float('inf')
    
    def initial_population(self):
        population = []
        for _ in range(self.population_size):
            route = list(range(self.num_cities))
            random.shuffle(route)
            population.append(route)
        return population
    
    def route_distance(self, route):
        distance = 0
        for i in range(self.num_cities):
            from_city = route[i]
            to_city = route[(i + 1) % self.num_cities]  
            distance += self.dist_matrix[from_city][to_city]
        return distance
    
    def rank_routes(self, population):
        fitness_results = {}
        for i, route in enumerate(population):
            distance = self.route_distance(route)
            fitness_results[i] = 1 / distance  
        
        return sorted([(population[i], fitness_results[i]) for i in range(len(population))], 
                      key=lambda x: x[1], reverse=True)
    
    def selection(self, ranked_population):
        selection_results = []
        
        for i in range(self.elite_size):
            selection_results.append(ranked_population[i][0])

        df = np.array([item[1] for item in ranked_population])
        df = df / df.sum()  
        cumulative_sum = np.cumsum(df)
        
        for _ in range(self.population_size - self.elite_size):
            pick = random.random()
            for i, cs in enumerate(cumulative_sum):
                if pick <= cs:
                    selection_results.append(ranked_population[i][0])
                    break
        
        return selection_results
    
    def create_population(self, mating_pool):
        children = []
        
        for i in range(self.elite_size):
            children.append(mating_pool[i])
        
        pool = random.sample(mating_pool, len(mating_pool)) 
        
        for i in range(self.population_size - self.elite_size):
            child = self.crossover(pool[i], pool[len(mating_pool) - i - 1])
            children.append(child)
        
        return children
    
    def crossover(self, parent1, parent2):
        child = [-1] * self.num_cities
        
        start, end = sorted(random.sample(range(self.num_cities), 2))

        for i in range(start, end + 1):
            child[i] = parent1[i]
        
        pointer = (end + 1) % self.num_cities
        parent2_pointer = (end + 1) % self.num_cities
        
        while -1 in child:
            if parent2[parent2_pointer] not in child:
                child[pointer] = parent2[parent2_pointer]
                pointer = (pointer + 1) % self.num_cities
            
            parent2_pointer = (parent2_pointer + 1) % self.num_cities
        
        return child
    
    def mutate(self, individual):
        for i in range(self.num_cities):
            if random.random() < self.mutation_rate:
                j = random.randint(0, self.num_cities - 1)
                individual[i], individual[j] = individual[j], individual[i]
        return individual
    
    def mutate_population(self, population):
        mutated_pop = []
        
        for i in range(self.elite_size):
            mutated_pop.append(population[i])
        
        for i in range(self.elite_size, self.population_size):
            mutated = self.mutate(population[i])
            mutated_pop.append(mutated)
        
        return mutated_pop
    
    def next_generation(self):
        ranked_pop = self.rank_routes(self.population)
        
        current_best_route = ranked_pop[0][0]
        current_best_distance = self.route_distance(current_best_route)

        if current_best_distance < self.best_distance:
            self.best_route = current_best_route.copy()
            self.best_distance = current_best_distance
        
        self.best_distance_history.append(current_best_distance)
        avg_distance = sum(1/item[1] for item in ranked_pop) / len(ranked_pop)
        self.avg_distance_history.append(avg_distance)
        
        selection_results = self.selection(ranked_pop)
        
        children = self.create_population(selection_results)
        
        self.population = self.mutate_population(children)
        
        return current_best_distance
    
    def run(self):
        start_time = time.time()
        
        print("Iniciando algoritmo genético...")
        print(f"Distância inicial da melhor rota: {1/self.ranked_population[0][1]:.2f}")
        
        for i in range(self.generations):
            current_best_distance = self.next_generation()
            
            if (i + 1) % 10 == 0 or i == 0 or i == self.generations - 1:
                print(f"Geração {i+1}/{self.generations}: Melhor distância = {current_best_distance:.2f}")
        
        execution_time = time.time() - start_time
        
        print(f"\nDistância final da melhor rota: {self.best_distance:.2f}")
        print(f"Melhoria: {(1 - self.best_distance/self.initial_distance) * 100:.2f}%")
        print(f"Tempo de execução: {execution_time:.2f} segundos")
        
        return {
            'best_route': self.best_route,
            'best_distance': self.best_distance,
            'execution_time': execution_time,
            'best_distance_history': self.best_distance_history,
            'avg_distance_history': self.avg_distance_history
        }
    
    def plot_progress(self):
        plt.figure(figsize=(12, 6))
        generations = range(1, len(self.best_distance_history) + 1)
        
        plt.plot(generations, self.best_distance_history, 'b-', label='Melhor distância')
        plt.plot(generations, self.avg_distance_history, 'r-', label='Distância média')
        
        plt.title('Evolução da distância ao longo das gerações')
        plt.xlabel('Geração')
        plt.ylabel('Distância')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_route(self, cities_coords=None):
        if cities_coords is None or self.best_route is None:
            return
        
        plt.figure(figsize=(10, 8))
        
        x = [cities_coords[i][0] for i in range(len(cities_coords))]
        y = [cities_coords[i][1] for i in range(len(cities_coords))]
        
        plt.scatter(x, y, c='red', s=100)
        
        for i in range(self.num_cities):
            from_city = self.best_route[i]
            to_city = self.best_route[(i + 1) % self.num_cities]
            plt.plot([cities_coords[from_city][0], cities_coords[to_city][0]],
                     [cities_coords[from_city][1], cities_coords[to_city][1]], 'b-')
        
        for i in range(self.num_cities):
            plt.annotate(f'{i}', (x[i], y[i]), xytext=(5, 5), textcoords='offset points')
        
        plt.title(f'Melhor rota encontrada (Distância: {self.best_distance:.2f})')
        plt.grid(True)
        plt.axis('equal')
        plt.show()

def generate_random_cities(num_cities, seed=42):
    np.random.seed(seed)
    return np.random.rand(num_cities, 2) * 100

def calculate_distance_matrix(cities):
    num_cities = len(cities)
    dist_matrix = np.zeros((num_cities, num_cities))
    
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                dist_matrix[i][j] = np.sqrt(((cities[i][0] - cities[j][0]) ** 2) + 
                                           ((cities[i][1] - cities[j][1]) ** 2))
            else:
                dist_matrix[i][j] = 0
    
    return dist_matrix

def run_example():
    num_cities = 1000
    population_size = 100
    elite_size = 20
    mutation_rate = 0.01
    generations = 500
    
    cities = generate_random_cities(num_cities)
    dist_matrix = calculate_distance_matrix(cities)
    
    ga = GeneticAlgorithm(
        dist_matrix=dist_matrix,
        population_size=population_size,
        elite_size=elite_size,
        mutation_rate=mutation_rate,
        generations=generations
    )
    
    results = ga.run()
 
    ga.plot_progress()
    ga.plot_route(cities)
    
    return results

if __name__ == "__main__":
    run_example()