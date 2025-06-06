# DGTA-RL Route Planner

An AI-powered route planner that combines Deep Reinforcement Learning with real-world road navigation using OpenStreetMap data.

## 🌍 Features

* Visual route planning with Leaflet.js
* DGTA-RL model for solving Time-Dependent and Stochastic TSP
* OpenStreetMap-based realistic travel time calculation
* Animated route preview along real roads
* Built with FastAPI, Torch, Alpine.js, and Tailwind CSS

---

## 📦 Project Structure

```
DGTA-RL/
├── frontend/               # Leaflet + Alpine.js UI
│   └── index.html
├── backend/                # FastAPI backend
│   ├── serve.py
│   ├── model/
│   ├── osm_graph.py
│   └── dgta_rl.pt          # Trained DGTA model
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/DGTA-RL.git
cd DGTA-RL
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn serve:app --reload --host 0.0.0.0 --port 8000
```

### 3. Frontend Setup

```bash
cd ../frontend
python -m http.server 8080
```

Then open [http://localhost:8080](http://localhost:8080) in your browser.

---

## 🔍 How it Works

1. You click on map to add points
2. Frontend sends coordinates to backend
3. Backend loads OSM graph for Tehran (or other)
4. Computes travel-time matrix + solves tour using DGTA model
5. Returns tour order + realistic route path
6. Frontend draws route, labels stops, and animates travel

---

## 🛠 Dependencies

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

## 🌐 Deployment Notes

* Make sure ports 8000 (API) and 8080 (frontend) are open
* Replace `localhost` with actual domain or public IP when deploying
* Consider Dockerizing for cloud or VPS deployment

---

## 📌 Todo

* ✅ Animation
* 🔲 Pause/Resume
* 🔲 Speed control
* 🔲 Multi-city support
* 🔲 Export to GPX/KML

---

## 🧠 Credits

This project is based on the "DGTA-RL" paper (2025) and integrates OpenStreetMap via `osmnx`. Frontend powered by Leaflet and Alpine.js.

> Built for educational and demonstration purposes.
