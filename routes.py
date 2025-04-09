import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Define uma semente aleatória para garantir a reprodutibilidade dos resultados
np.random.seed(42)

# Configura o estilo do matplotlib para gráficos mais atraentes
plt.style.use('seaborn-v0_8-whitegrid')

# Cidades do trajeto Porto Alegre até São Luís (Maranhão)
# Selecionamos cidades importantes que formam um trajeto possível
cities = ["Porto Alegre", "Florianópolis", "Curitiba", "São Paulo", 
          "Rio de Janeiro", "Belo Horizonte", "Brasília", "Palmas", 
          "Imperatriz", "São Luís"]

# Definir a base de dados de tráfego por hora do dia (0 a 23)
horas = np.arange(0, 24, 1)

# Cria função para simular tráfego mais realista
def simular_trafego(hora):
    # Base de tráfego (valor médio)
    base = 1.0
    
    # Pico da manhã (7-9h)
    if 7 <= hora <= 9:
        base += 0.8 * np.sin((hora - 7) * np.pi / 2)
    
    # Pico da tarde (17-19h)
    elif 17 <= hora <= 19:
        base += 0.9 * np.sin((hora - 17) * np.pi / 2)
    
    # Noite (22-5h) - tráfego reduzido
    elif hora >= 22 or hora <= 5:
        base -= 0.4
    
    # Adiciona ruído aleatório
    noise = np.random.normal(0, 0.1)
    
    return base + noise

# Gerar dados de tráfego para cada hora
tempo_viagem = np.array([simular_trafego(h) for h in horas])

# Normaliza os dados de tráfego para treinamento
scaler = MinMaxScaler()
tempo_viagem_scaled = scaler.fit_transform(tempo_viagem.reshape(-1, 1))

# Cria DataFrame
df = pd.DataFrame({"hora": horas, "tempo_viagem": tempo_viagem_scaled.flatten()})

# Função para criar sequências de treinamento
def create_sequences(data, seq_length=3):
    X, y = [], []
    # Cria sequências de entrada (X) e saída (y)
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Define o comprimento da sequência
seq_length = 3

# Gera os dados de treinamento
X, y = create_sequences(tempo_viagem_scaled, seq_length)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Define e treina o modelo LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, batch_size=8, verbose=1)

# Realiza previsões
predictions = model.predict(X)

# Converte previsões para a escala original
predicted_traffic = scaler.inverse_transform(predictions)

# Cria grafo
G = nx.Graph()

# Cria dicionário com coordenadas geográficas (latitude, longitude) das cidades
cities_pos = {
    'Porto Alegre': (-30.0346, -51.2177),
    'Florianópolis': (-27.5969, -48.5495),
    'Curitiba': (-25.4297, -49.2719),
    'São Paulo': (-23.5505, -46.6333),
    'Rio de Janeiro': (-22.9068, -43.1729),
    'Belo Horizonte': (-19.9167, -43.9345),
    'Brasília': (-15.7801, -47.9292),
    'Palmas': (-10.2491, -48.3243),
    'Imperatriz': (-5.5264, -47.4712),
    'São Luís': (-2.5307, -44.3068)
}

# Adiciona as cidades como nós
for city in cities:
    G.add_node(city)

# Definição das distâncias aproximadas entre cidades (km)
# e uso das previsões de tráfego para ajustar os tempos de viagem
distances = [
    ("Porto Alegre", "Florianópolis", 460),
    ("Florianópolis", "Curitiba", 300),
    ("Curitiba", "São Paulo", 408),
    ("São Paulo", "Rio de Janeiro", 429),
    ("Rio de Janeiro", "Belo Horizonte", 442),
    ("Belo Horizonte", "Brasília", 716),
    ("Brasília", "Palmas", 973),
    ("Palmas", "Imperatriz", 623),
    ("Imperatriz", "São Luís", 637),
    # Algumas conexões alternativas
    ("São Paulo", "Belo Horizonte", 586),
    ("Brasília", "Imperatriz", 1311),
    ("Curitiba", "Belo Horizonte", 1004),
    ("Belo Horizonte", "Palmas", 1690),
]

# Converte distâncias em tempo de viagem (assumindo velocidade média de 80 km/h)
# e adiciona o impacto do tráfego
edges = []
for i, (city1, city2, dist) in enumerate(distances):
    # Usa previsões diferentes para cada conexão, reutilizando em ciclo se necessário
    traffic_index = i % len(predicted_traffic)
    
    # Calcula o tempo base em horas (distância / velocidade média)
    base_time = dist / 80.0
    
    # Adiciona o impacto do tráfego (entre 0% e 50% mais tempo)
    traffic_factor = 1 + 0.5 * predicted_traffic[traffic_index][0]
    
    # Tempo total em horas
    travel_time = base_time * traffic_factor
    
    # Adiciona à lista de arestas
    edges.append((city1, city2, travel_time))

# Adiciona as arestas ponderadas ao grafo
G.add_weighted_edges_from(edges)

# Define origem e destino
source, target = "Porto Alegre", "São Luís"

# Encontra caminho mais curto
shortest_path = nx.shortest_path(G, source=source, target=target, weight='weight')

# Calcula o tempo total de viagem
total_time = 0
path_edges = list(zip(shortest_path[:-1], shortest_path[1:]))
for u, v in path_edges:
    total_time += G[u][v]['weight']

# Seleciona 3 cidades intermediárias mais relevantes
intermediary_cities = shortest_path[1:-1]
if len(intermediary_cities) > 3:
    # Estratégia: pegar cidades em pontos distribuídos do trajeto
    indices = np.linspace(0, len(intermediary_cities)-1, 3).astype(int)
    key_cities = [intermediary_cities[i] for i in indices]
else:
    key_cities = intermediary_cities

# Exibe resultados no console
print(f"Melhor rota de {source} para {target}: {shortest_path}")
print(f"Cidades intermediárias mais importantes: {key_cities}")
print(f"Tempo total estimado: {total_time:.1f} horas ({total_time/24:.1f} dias de viagem)")

# =======================================================
# VISUALIZAÇÃO MELHORADA DOS RESULTADOS
# =======================================================

# Cria uma figura maior com título principal
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle('Sistema de Navegação: Porto Alegre até Maranhão', fontsize=18, y=0.98)

# Gráfico 1: Previsão de tráfego
axes[0].plot(horas[seq_length:], scaler.inverse_transform(y.reshape(-1,1)), 
             label="Tráfego Real", linewidth=2, color='blue')
axes[0].plot(horas[seq_length:], predicted_traffic, 
             label="Tráfego Previsto", linestyle="dashed", linewidth=2, color='red')
axes[0].set_xlabel("Hora do Dia", fontsize=12)
axes[0].set_ylabel("Índice de Tráfego", fontsize=12)
axes[0].legend(fontsize=10)
axes[0].set_title("Previsão de Tráfego ao Longo do Dia", fontsize=14)
axes[0].grid(True)

# Adiciona sombreamento para destacar picos de tráfego
axes[0].axvspan(7, 9, alpha=0.2, color='orange', label='Pico da Manhã')
axes[0].axvspan(17, 19, alpha=0.2, color='orange', label='Pico da Tarde')
axes[0].text(8, 0.2, "Pico da Manhã", ha='center', fontsize=10)
axes[0].text(18, 0.2, "Pico da Tarde", ha='center', fontsize=10)

# Gráfico 2: Grafo de rotas
# pos = nx.spring_layout(G, seed=42, k=.5)  # k controla o espaçamento entre os nós
pos = cities_pos
pos = {city: (lat, long) for (city, (long, lat)) in cities_pos.items()}

# Destacar o caminho mais curto
edge_colors = ['red' if (u, v) in path_edges or (v, u) in path_edges else 'gray' for u, v in G.edges()]
edge_widths = [3 if (u, v) in path_edges or (v, u) in path_edges else 1 for u, v in G.edges()]

# Desenha o grafo
nx.draw(G, pos, with_labels=True, node_size=3000, 
        node_color=['gold' if node == source or node == target else 
                   'lightgreen' if node in key_cities else 
                   'lightblue' if node in shortest_path else 
                   'white' for node in G.nodes()],
        font_size=10, font_weight='bold', 
        edge_color=edge_colors, width=edge_widths, ax=axes[1])

# Adiciona os pesos das arestas (tempo em horas)
edge_labels = {(u, v): f"{d['weight']:.1f}h" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=axes[1])

# Adiciona legenda para o mapa
legend_elements = [
    Patch(facecolor='gold', edgecolor='black', label='Origem/Destino'),
    Patch(facecolor='lightgreen', edgecolor='black', label='Cidades-chave'),
    Patch(facecolor='lightblue', edgecolor='black', label='Na rota'),
    Patch(facecolor='white', edgecolor='black', label='Fora da rota'),
    Line2D([0], [0], color='red', lw=3, label='Rota selecionada'),
    Line2D([0], [0], color='gray', lw=1, label='Conexões alternativas')
]

axes[1].legend(handles=legend_elements, loc='lower right', fontsize=10)
axes[1].set_title("Rota Otimizada de Porto Alegre até São Luís (Maranhão)", fontsize=14)

# Ajusta o layout
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Espaço para o título principal

# Salva os gráficos em vários formatos
output_dir = "./"  # Diretório onde os gráficos serão salvos

# Salva o gráfico completo (com os dois subplots)
plt.savefig(f'{output_dir}rota_porto_alegre_maranhao.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}rota_porto_alegre_maranhao.pdf', bbox_inches='tight')
plt.savefig(f'{output_dir}rota_porto_alegre_maranhao.svg', bbox_inches='tight')

# ===========================================================
# SALVA GRÁFICOS INDIVIDUAIS EM ARQUIVOS SEPARADOS
# ===========================================================

# Salva o gráfico de previsão de tráfego separadamente
fig_traffic = plt.figure(figsize=(12, 8))
plt.plot(horas[seq_length:], scaler.inverse_transform(y.reshape(-1,1)), 
         label="Tráfego Real", linewidth=2, color='blue')
plt.plot(horas[seq_length:], predicted_traffic, 
         label="Tráfego Previsto", linestyle="dashed", linewidth=2, color='red')
plt.axvspan(7, 9, alpha=0.2, color='orange')
plt.axvspan(17, 19, alpha=0.2, color='orange')
plt.text(8, 0.2, "Pico da Manhã", ha='center', fontsize=10)
plt.text(18, 0.2, "Pico da Tarde", ha='center', fontsize=10)
plt.xlabel("Hora do Dia", fontsize=12)
plt.ylabel("Índice de Tráfego", fontsize=12)
plt.legend(fontsize=10)
plt.title("Previsão de Tráfego ao Longo do Dia", fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{output_dir}previsao_trafego.png', dpi=300, bbox_inches='tight')
plt.close(fig_traffic)

# Salva o mapa de rota separadamente
fig_route = plt.figure(figsize=(14, 12))
nx.draw(G, pos, with_labels=True, node_size=3000, 
        node_color=['gold' if node == source or node == target else 
                   'lightgreen' if node in key_cities else 
                   'lightblue' if node in shortest_path else 
                   'white' for node in G.nodes()],
        font_size=12, font_weight='bold', 
        edge_color=edge_colors, width=edge_widths)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

# Adiciona a legenda
plt.legend(handles=legend_elements, loc='lower right', fontsize=12)
plt.title("Rota Otimizada de Porto Alegre até São Luís (Maranhão)", fontsize=16)
plt.tight_layout()
plt.savefig(f'{output_dir}mapa_rota.png', dpi=300, bbox_inches='tight')
plt.close(fig_route)

# Cria um arquivo de texto com os detalhes da rota calculada
with open(f'{output_dir}detalhes_rota.txt', 'w') as f:
    f.write(f"DETALHES DA ROTA CALCULADA\n")
    f.write(f"==========================\n\n")
    f.write(f"Origem: {source}\n")
    f.write(f"Destino: {target}\n\n")
    f.write(f"Rota completa: {' -> '.join(shortest_path)}\n\n")
    f.write(f"Cidades-chave selecionadas: {', '.join(key_cities)}\n\n")
    f.write(f"Tempo total estimado: {total_time:.1f} horas ({total_time/24:.1f} dias)\n\n")
    f.write(f"Detalhes de cada trecho:\n")
    
    for i in range(len(shortest_path)-1):
        city1, city2 = shortest_path[i], shortest_path[i+1]
        time = G[city1][city2]['weight']
        f.write(f"  {city1} -> {city2}: {time:.1f} horas\n")

# Exibe o gráfico na tela (se estiver executando interativamente)
plt.figure(figsize=(20, 10))
plt.figtext(0.5, 0.01, "Gráficos salvos com sucesso! Verifique os arquivos gerados.", 
           ha='center', fontsize=14, bbox=dict(facecolor='yellow', alpha=0.5))
plt.axis('off')
plt.tight_layout()
plt.show()