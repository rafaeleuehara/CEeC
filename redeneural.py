import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def activation(vector):
	return 1/(1+np.exp((-1)*vector))	# Função de ativação sigmóide

df = pd.read_csv("output.csv",header=None)
df = df.sample(frac=1).reset_index(drop=True) 	# Embaralha os dados

df.columns = [
'ID','Diagnosis','radius1','texture1','perimeter1','area1','smoothness1',
'compactness1','concavity1','concave_points1','symmetry1','fractal_dimension1',
'radius2','texture2','perimeter2','area2','smoothness2','compactness2',
'concavity2','concave_points2','symmetry2','fractal_dimension2',
'radius3','texture3','perimeter3','area3','smoothness3','compactness3',
'concavity3','concave_points3','symmetry3','fractal_dimension3'
] 	# Como este dataset não possui header, foi adicionado manualmente com o pandas

X_data = df.drop(['ID','Diagnosis'],axis=1) # Cria o X_data sem as colunas de ID e Diagnosis
Diagnosis = df['Diagnosis']

Y_data = np.zeros([569,2])

for i in range(569): 	# Loop para classificação entre tumor maligno (0,1) e benigno (1,0)
	if Diagnosis[i] == 'B':
		Y_data[i,0] = 1
	elif Diagnosis[i] == 'M':
		Y_data[i,1] = 1

X_data = np.append(X_data,np.ones((569,1)),axis=1) 	 # Criação de uma coluna para o bias
X_data = (X_data - X_data.mean()) / X_data.std() 	# Normaliza os dados para evitar overflow tendo valores entre -3 e 3

NN = np.random.rand(2,31)-0.5	 # Pesos aleatórios para a primeira época

X_train = X_data[:427,:]	 # 427 instâncias são usadas para o treino
Y_train = Y_data[:427,:]

X_test = X_data[427:,:] 	# 142 instâncias são usadas para o teste
Y_test = Y_data[427:,:]

num_epochs = 1000 	# Define o número de épocas
learning_rate = 0.01 	# Define o learning rate
loss_vector = [] 	# Vetor usado para plotar um gráfico de relação entre número de épocas pelo erro

for epoch in range(num_epochs): 	# Treino da rede neural
	for idx in range(X_train.shape[0]):
		input_instance = X_data[idx,:]
		output = activation(np.matmul(input_instance,NN.T))
		error = output - Y_train[idx,:]
		
		for i in range(2):
			for j in range(31):
				NN[i,j] -= learning_rate*2*error[i]*output[i]*(1-output[i])*input_instance[j] 	# Backpropagation
		
	test_outputs = activation(np.matmul(X_test,NN.T)) 	# Resultado de cada época
	test_errors = test_outputs - Y_test		# Cálculo do erro a cada época
		
	MSE = 0	
		
	for i in range(test_errors.shape[0]):	# Calcula o MSE
		for j in range(test_errors.shape[1]):
			MSE += test_errors[i,j]**2
   
	MSE /= test_errors.shape[0]*test_errors.shape[1]
	RMSE = np.sqrt(MSE)		# Usando RMSE como métrica de regressão
			
	loss_vector.append(RMSE) 	# Coloca o valor do RMSE numa matriz para renderizar o gráfico

CM = np.zeros([2,2]) 	# Cria a matriz confusão

for instance in range(X_test.shape[0]): 	# Coloca os resultados numa matriz confusão
	input_instance = X_test[instance,:]
	output = activation(np.matmul(input_instance,NN.T))
	prediction = np.argmax(output)
	data = np.argmax(Y_test[instance,:])
	
	CM[data,prediction] += 1

# Criação de variáveis para a matriz de confusão a fim de deixar o código mais organizado e legível
true_negative = CM[0,0]
true_positive = CM[1,1]
false_negative = CM[1,0]
false_positive = CM[0,1]

# Calcula a acurácia global e precisão, recall e F1-Score para cada uma das classes (maligna e benigna)
global_precision = (true_negative + true_positive) / (true_positive + true_negative + false_negative + false_positive)

precision_classe1 = true_negative / (true_negative + false_negative)
recall_classe1 = true_negative / (true_negative + false_positive)
f1_score_classe1 = 2 * ((precision_classe1 * recall_classe1) / (precision_classe1 + recall_classe1))

precision_classe2 = true_positive / (true_positive + false_positive)
recall_classe2 = true_positive / (true_positive + false_negative)
f1_score_classe2 = 2 * ((precision_classe2 * recall_classe2) / (precision_classe2 + recall_classe2))

# Printa os resultados da rede neural
print(f"Total instâncias: {true_positive + true_negative + false_negative + false_positive:.0f}")

print(f"""
{CM}
""")

print(f"""Acurácia Global: {global_precision:.2f}
      
Classe Benigna ({CM[0,0]+CM[0,1]:.0f}): 
Precisão: {precision_classe1:.2f}
Recall: {recall_classe1:.2f}
F1 Score: {f1_score_classe1:.2f}

Classe Maligna ({CM[1,1]+CM[1,0]:.0f}):
Precisão: {precision_classe2:.2f}
Recall: {recall_classe2:.2f}
F1 Score: {f1_score_classe2:.2f}
""")

# Renderiza o gráfico do erro pelas épocas
plt.figure(dpi=100, figsize=(8,6))
plt.plot(range(num_epochs),loss_vector)
plt.ylabel('Erro')
plt.xlabel('Épocas')
plt.title('Gráfico do erro ao longo das épocas')
plt.show()