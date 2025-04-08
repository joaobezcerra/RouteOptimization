import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define uma semente aleatória para garantir a reprodutibilidade dos resultados.
np.random.seed(42)

# Cria um array representando as horas do dia (de 0 a 23).
horas = np.arange(0, 24, 1)

# Simula o tempo de viagem ao longo do dia usando uma função senoidal (para representar padrões cíclicos)
# e adiciona ruído normal para tornar os dados mais realistas.
tempo_viagem = np.sin(horas / 24 * 2 * np.pi) + np.random.normal(0, 0.1, len(horas))

# Inicializa um MinMaxScaler para normalizar os dados de tempo de viagem para o intervalo [0, 1].
# Isso é importante para o treinamento de redes neurais.
scaler = MinMaxScaler()

# Ajusta o scaler aos dados de tempo de viagem e os transforma. O reshape(-1, 1) é necessário
# porque o scaler espera uma matriz 2D.
tempo_viagem_scaled = scaler.fit_transform(tempo_viagem.reshape(-1, 1))

# Cria um DataFrame do pandas para organizar os dados de hora e tempo de viagem escalonado.
df = pd.DataFrame({"hora": horas, "tempo_viagem": tempo_viagem_scaled.flatten()})

# Define uma função para criar sequências de dados para treinamento de modelos de séries temporais.
# A função recebe os dados e o comprimento da sequência desejada (seq_length).
def create_sequences(data, seq_length=3):
    X, y = [], []
    # Itera pelos dados, criando janelas de tamanho seq_length como entrada (X)
    # e o valor seguinte como a saída esperada (y).
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    # Converte as listas de entrada e saída em arrays numpy.
    return np.array(X), np.array(y)

# Define o comprimento da sequência para 3 (usaremos as últimas 3 horas para prever a próxima).
seq_length = 3

# Chama a função create_sequences para gerar os dados de entrada (X) e saída (y) para o modelo LSTM.
X, y = create_sequences(tempo_viagem_scaled, seq_length)

# Redimensiona a entrada X para o formato esperado por uma camada LSTM:
# (número de amostras, número de passos de tempo, número de features).
# Neste caso, temos 1 feature (o tempo de viagem escalonado).
X = X.reshape(X.shape[0], X.shape[1], 1)

# Define a arquitetura do modelo de rede neural sequencial usando Keras.
model = Sequential([
    # Primeira camada LSTM com 50 unidades, retornando sequências (para a próxima camada LSTM).
    # Define o formato de entrada para a primeira camada: sequências de comprimento seq_length com 1 feature.
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    # Segunda camada LSTM com 50 unidades. Por padrão, retorna apenas a última saída.
    LSTM(50),
    # Camada densa (totalmente conectada) com 1 unidade para a previsão do tempo de viagem.
    Dense(1)
])

# Compila o modelo, definindo o otimizador ('adam' é um otimizador comum),
# a função de perda ('mse' - erro quadrático médio, adequado para regressão) e as métricas (não especificadas aqui).
model.compile(optimizer='adam', loss='mse')

# Treina o modelo com os dados de entrada X e saída y.
# epochs: número de vezes que o modelo percorrerá todo o conjunto de treinamento.
# batch_size: número de amostras a serem processadas por vez durante o treinamento.
# verbose=1: exibe informações sobre o progresso do treinamento.
model.fit(X, y, epochs=50, batch_size=8, verbose=1)

# Faz previsões usando o modelo treinado com os dados de entrada X.
predictions = model.predict(X)

# Inverte a escala das previsões para obter os valores de tempo de viagem na escala original (em minutos).
predicted_traffic = scaler.inverse_transform(predictions)

# Cria um grafo não direcionado usando a biblioteca NetworkX para representar a rede de rotas.
G = nx.Graph()

# Define uma lista de cidades que farão parte da rede.
cities = ["A", "B", "C", "D", "E"]

# Adiciona cada cidade como um nó no grafo.
for city in cities:
    G.add_node(city)

# Define as conexões (arestas) entre as cidades e seus respectivos pesos.
# Os pesos representam o tempo de viagem entre as cidades, e são dinamicamente ajustados
# adicionando a previsão de tráfego (a primeira previsão para a primeira aresta, a segunda para a segunda, etc.).
edges = [
    ("A", "B", 10 + predicted_traffic[0][0]),
    ("B", "C", 15 + predicted_traffic[1][0]),
    ("C", "D", 5 + predicted_traffic[2][0]),
    ("D", "E", 8 + predicted_traffic[3][0]),
    ("A", "D", 20 + predicted_traffic[4][0])
]

# Adiciona as arestas ponderadas ao grafo. O argumento 'weight' armazena o peso de cada aresta.
G.add_weighted_edges_from(edges)

# Define a cidade de origem e a cidade de destino para encontrar o caminho mais curto.
source, target = "A", "E"

# Usa o algoritmo de Dijkstra (implementado no NetworkX) para encontrar o caminho mais curto
# entre a origem e o destino, considerando o peso das arestas ('weight').
shortest_path = nx.shortest_path(G, source=source, target=target, weight='weight')

# Imprime o caminho mais curto encontrado.
print(f"Melhor rota de {source} para {target}: {shortest_path}")

# Cria uma figura e um conjunto de subplots para visualizar os resultados.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de previsão de tráfego (primeiro subplot)
# Plota os valores reais do tempo de viagem (após a sequência inicial de 3 horas).
axes[0].plot(horas[seq_length:], scaler.inverse_transform(y.reshape(-1,1)), label="Real")
# Plota os valores previstos do tempo de viagem (correspondentes aos valores reais plotados).
axes[0].plot(horas[seq_length:], predicted_traffic, label="Previsto", linestyle="dashed")
# Define o rótulo do eixo x.
axes[0].set_xlabel("Hora do Dia")
# Define o rótulo do eixo y.
axes[0].set_ylabel("Tempo de Viagem (Min)")
# Adiciona uma legenda ao gráfico.
axes[0].legend()
# Define o título do subplot.
axes[0].set_title("Previsão de Tráfego")

# Layout para o grafo da rede de rotas (segundo subplot)
# Usa o algoritmo de layout de primavera para posicionar os nós do grafo de forma visualmente agradável.
pos = nx.spring_layout(G)
# Obtém os pesos das arestas do grafo.
labels = nx.get_edge_attributes(G, 'weight')
# Desenha o grafo com os nós rotulados, tamanho e cor dos nós, e tamanho da fonte dos rótulos.
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=12, ax=axes[1])
# Desenha os rótulos das arestas (os tempos de viagem previstos) no grafo, formatando-os para exibir uma casa decimal.
nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.1f} min" for k, v in labels.items()}, ax=axes[1])
# Define o título do subplot.
axes[1].set_title("Rede de Rotas com Previsão de Tráfego")

# Ajusta o layout dos subplots para evitar sobreposição.
plt.tight_layout()
# Exibe a figura com os dois subplots.
plt.show()