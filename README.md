# Otimização de Rotas com LSTM e Grafos

![Tráfego e Rotas](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Dijkstra_Animation.gif/500px-Dijkstra_Animation.gif)

# Sobre o Projeto
Este projeto usa **Redes Neurais LSTM** e **Grafos** para prever o tempo de viagem ao longo do dia e otimizar rotas. O objetivo é criar um modelo que analisa padrões de tráfego e sugere o caminho mais rápido entre pontos de interesse.

# Funcionalidades
- Simulação de dados de tráfego baseados em variações diárias.<br>
- Treinamento de um modelo **LSTM** para previsão de tempo de viagem.<br>
- Construção de um **Grafo Ponderado** representando as rotas.<br>
- Aplicação do **Algoritmo de Dijkstra** para encontrar a melhor rota.<br>
- Visualização gráfica das previsões e da rede de rotas.

# Tecnologias Utilizadas
- **Python (3.9.21)**
- **NumPy, Pandas** (Manipulação de dados)
- **Matplotlib** (Visualização)
- **NetworkX** (Grafos e Dijkstra)
- **Scikit-learn** (Normalização dos dados)
- **TensorFlow/Keras** (Modelo LSTM)

# Executar o Código
```bash
python otimizacao_rotas.py
```

# Visualizações Geradas
- **Gráfico de previsão de tráfego**
- **Mapa de rotas otimizadas com tempos de viagem**
# RouteOptimization
