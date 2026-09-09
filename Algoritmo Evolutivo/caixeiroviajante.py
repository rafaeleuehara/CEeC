import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

start = time.perf_counter()

# Cálculo do fitness	
def calc_fitness(solution, dist_matrix):
	dist = 0
	for i in range(dist_matrix.shape[0] - 1):
		dist += dist_matrix[solution[i],solution[i+1]]
	return dist

df = pd.read_csv("tsp_dataset.csv")

# Vetores para o gráfico
best_history = []
avg_history = []

# Escolha da instância do carteiro viajante
# IDs das instâncias escolhidas para os testes, números em parênteses representam o número de cidades na instância
# Pequenos: 1 (37), 10 (36), 11 (31), 17 (34)
# Médios: 5 (42), 6 (43), 7 (65), 9 (44)
# Grandes: 2 (98), 3 (83), 4 (92), 8 (73)
line = 1

# Criação dos vetores 
num_cities = df['num_cities'][line]
dist_matrix = np.matrix(df['distance_matrix'][line])
dist_matrix = np.reshape(dist_matrix,(num_cities,num_cities))

# Criação do tamanho da população
popsize = 300
population = np.zeros((popsize, num_cities))

# Populando o vetor população com uma sequência aleatória de caminhos por indivíduo
for i in range(popsize):
	population[i,:] = np.random.permutation(num_cities)

# Criação dos parâmetros do laço
num_epochs = 200
survival_rate = 0.5
mutation_prob = 0.05

# Criação de variáveis para análise resultados
best_solution = None
best_distance = np.inf

# Laço principal de seleção/reprodução/mutação
for epoch in range(num_epochs):
    
    # Criação do vetor fitness para cada indivíduo
	fitness = np.zeros(population.shape[0])
	
 	# Cálculo do fitness
	for i in range(population.shape[0]):
		fitness[i] = calc_fitness(population[i,:].astype(int),dist_matrix)
    
    # Laço para calcular a melhor solução e sua distância total
	current_best = np.argmin(fitness) # Pega a instância com menor fitness
    
	if fitness[current_best] < best_distance:
		best_distance = fitness[current_best] # Extrai o dist da instância
		best_solution = population[current_best].astype(int).copy() # Extrai o vetor solução da instância
        
	# Coloca o fitness médio e melhor fitness num vetor afim de plotar o gráfico ao final do algoritmo
	best_history.append(np.min(fitness))
	avg_history.append(np.mean(fitness))
    
    # Laço para seleção (Torneio)
	while population.shape[0] > survival_rate*popsize:
		p1 = np.random.randint(0,population.shape[0])
		p2 = np.random.randint(0,population.shape[0])
  
		# Evita que ambos os indivíduos selecionados seja o mesmo
		while p2 == p1:
			p2 = np.random.randint(0,population.shape[0])
		
		if fitness[p1] < fitness[p2]:
			population = np.delete(population,p2,axis=0)
			fitness = np.delete(fitness,p2,axis=0)
		else:
			population = np.delete(population,p1,axis=0)
			fitness = np.delete(fitness,p1,axis=0)
	
	# Laço para reprodução
	while population.shape[0] < popsize:
		p1 = np.random.randint(0,population.shape[0])
		p2 = np.random.randint(0,population.shape[0])
  
		# Evita que ambos os indivíduos selecionados seja o mesmo
		while p2 == p1:
			p2 = np.random.randint(0,population.shape[0])
			
		xo_point = np.random.randint(population.shape[1]) # Seleciona um ponto aleatório para fazer o crossover point
		offspring = population[p1,:xo_point].copy() # Cria o indivíduo filho com parte selecionada do pai 1
		
		# Insere o restante do genótipo faltante do pai 2
		for elem in population[p2,:]:
			if elem not in offspring:
				offspring = np.append(offspring,[elem],axis=0)
		
		# Bloco para mutação (Mudança de ordem)
		if np.random.rand() < mutation_prob:
			pos1 = np.random.randint(num_cities)
			pos2 = np.random.randint(num_cities)
			offspring[pos1], offspring[pos2] = offspring[pos2], offspring[pos1]
		
		# Insere o filho na população
		offspring = np.reshape(offspring, [1,num_cities])
		population = np.append(population,offspring,axis=0)

end = time.perf_counter()

print(f"Tempo: {end-start:.2f} segundos")

print(f"Melhor rota encontrada: {best_solution}")

print(f"Distância total: {best_distance:.2f}")

plt.plot(best_history, label='Melhor Fitness')
plt.plot(avg_history, label='Fitness Medio')

plt.xlabel('Épocas')
plt.ylabel('Fitness')
plt.title('Evolução do fitness ao longo das gerações')
plt.legend()
plt.grid(True)
plt.show()
