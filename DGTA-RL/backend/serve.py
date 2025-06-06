# serve.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
from dgta import DGTA
from osm_graph import load_graph, get_nearest_nodes, build_travel_time_matrix
import networkx as nx
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and graph
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = DGTA().to(DEVICE)
model.load_state_dict(torch.load("dgta_rl.pt", map_location=DEVICE))
model.eval()
G = load_graph("Tehran, Iran")

class LocationRequest(BaseModel):
    locations: list[list[float]]  # [[lat, lon], ...]

@app.post("/solve-tour")
async def solve_tour(data: LocationRequest):
    coords = data.locations
    node_ids = get_nearest_nodes(G, coords)
    time_matrix = build_travel_time_matrix(G, node_ids)

    from env import DTSPTDS
    N = len(coords)
    env = DTSPTDS(N=N, T=12, device=DEVICE)
    env.coords = np.array(coords)
    env.u_hat = np.ones((12, N, N))

    def fixed_travel_time(i, j, t):
        return time_matrix[i][j]

    env._travel_time = fixed_travel_time
    env.reset()
    state = env._build_state()

    done = False
    tour = [0]
    while not done:
        with torch.no_grad():
            logits = model(state['coords'].unsqueeze(0),
                           state['t_idx'],
                           state['visited'].unsqueeze(0),
                           torch.tensor([[state['curr']]], device=DEVICE))
            action = torch.argmax(logits, dim=-1).item()
        tour.append(action)
        state, _, done, _ = env.step(action)

    # Reconstruct full road path
    full_route = []
    for i in range(len(tour) - 1):
        u = node_ids[tour[i]]
        v = node_ids[tour[i + 1]]
        try:
            path_nodes = nx.shortest_path(G, u, v, weight="travel_time")
            path_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in path_nodes]
            full_route.extend(path_coords)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    ordered_coords = [coords[i] for i in tour]
    return {"tour": tour, "path": ordered_coords, "route": full_route}

if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)
