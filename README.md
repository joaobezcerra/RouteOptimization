# Route Optimization with LSTM and Graphs

![Traffic and Routes](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Dijkstra_Animation.gif/500px-Dijkstra_Animation.gif)

# About the Project
This project uses **LSTM Neural Networks** and **Graphs** to predict travel times throughout the day and optimize routes. The goal is to create a model that analyzes traffic patterns and suggests the fastest route between points of interest.

# Features
- Simulation of traffic data based on daily variations.<br>
- Training of an **LSTM** model for travel time prediction.<br>
- Construction of a **Weighted Graph** representing the routes.<br>
- Application of the **Dijkstra Algorithm** to find the best route.<br>
- Graphical visualization of variations and the route network.

# Technologies Used
- **Python (3.9.21)**
- **NumPy, Pandas** (Data manipulation)
- **Matplotlib** (Visualization)
- **NetworkX** (Graphs and Dijkstra)
- **Scikit-learn** (Data normalization)
- **TensorFlow/Keras** (LSTM model)

# Run the Code
```bash
python routes.py
```

# Generated Visualizations
- **Traffic forecast graph**
- **Optimized route map with travel times**
