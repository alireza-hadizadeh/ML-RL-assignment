# 📘 DGTA-RL Route Planner - Full Project Documentation

This project implements a Deep Reinforcement Learning-based route planner that uses OpenStreetMap (OSM) for realistic routing and travel time computation. It supports road-following paths, animated route playback, and interactive UI features.

---

## 📁 Project Structure Overview

```
DGTA-RL/
├── backend/
│   ├── serve.py
│   ├── osm_graph.py
│   ├── model/
│   │   ├── dgta.py
│   │   ├── dynamic_encoder.py
│   │   ├── pointer_decoder.py
│   │   ├── dual_attention.py
│   │   └── positional_encoding.py
│   ├── env.py
│   └── dgta_rl.pt
├── frontend/
│   └── index.html
├── requirements.txt
└── README.md
```

---

## 🧠 Backend Files

### `serve.py`

* Main FastAPI server.
* Loads the trained DGTA-RL model.
* Accepts coordinates via POST `/solve-tour`.
* Computes the tour using the model and returns the tour order, path coordinates, and real-road-following route.
* Uses OSM graph from `osm_graph.py`.

### `osm_graph.py`

* Handles OpenStreetMap graph loading using `osmnx`.
* Defines:

  * `load_graph(city_name)`: loads and caches OSM road graph.
  * `get_nearest_nodes(G, coords)`: maps user input points to OSM node IDs.
  * `build_travel_time_matrix(G, node_ids)`: builds a NxN matrix of shortest travel times.

### `env.py`

* Simulates the environment for the DGTA-RL model (based on the DTSP-TDS problem).
* Handles time-dependent and stochastic travel time logic.
* Encapsulates the RL environment interface (`reset()`, `step()`, `_build_state()`).

### `model/dgta.py`

* The full DGTA neural network architecture.
* Integrates the following components:

  * `PositionalEncoding`
  * `DualAttentionBlock`
  * `DynamicEncoder`
  * `PointerDecoder`
* Defines `forward()` that computes logits for route decisions.

### `model/dual_attention.py`

* Contains the `DualAttentionBlock`.
* Applies spatial and temporal attention (multi-head), then residual feedforward processing.

### `model/positional_encoding.py`

* Implements standard sinusoidal positional encoding.
* Used to encode order into input coordinates/time data.

### `model/dynamic_encoder.py`

* Selects the relevant dynamic hidden states based on current time and node.
* Uses a gating mechanism to fuse temporal and spatial attention outputs.

### `model/pointer_decoder.py`

* Pointer network component.
* Uses attention over the encoded sequence to predict the next node.

### `dgta_rl.pt`

* Trained PyTorch model weights (DGTA-RL).
* Loaded into the backend on server start.

---

## 🌐 Frontend Files

### `index.html`

* Main user interface with interactive map (Leaflet.js).
* Uses:

  * **Leaflet.js**: map and marker UI
  * **Alpine.js**: lightweight reactive JS logic
  * **Tailwind CSS**: styling
* Key features:

  * Add points by clicking on the map
  * Solve tour via backend
  * Display stop order markers with labels
  * Draw animated route line using real OSM roads
  * Show estimated total travel time
  * Reset and Undo functionality

---

## 📦 requirements.txt

Contains all Python dependencies:

```txt
fastapi
uvicorn
torch
networkx
osmnx==2.0.3
numpy
scipy
```

---

## 🏁 Future Additions

* Add real-time traffic data via API or simulation.
* Add pause/play speed control to animation.
* Export path to GPX or KML for use in GPS devices.
* Add multi-city or user-selectable regions.

---

## 🧠 Summary

This system combines deep reinforcement learning and realistic road network routing to solve complex path planning problems with user-friendly visual tools. It supports learning-based optimization (DGTA-RL), OSM-based environment modeling, and interactive visualization of routes and timing.
