# osm_graph.py
import osmnx as ox
import networkx as nx
import numpy as np

# Load street graph (e.g. driving in Tehran)
def load_graph(place="Tehran, Iran"):
    print(f"Loading OSM graph for: {place}")
    G = ox.graph_from_place(place, network_type="drive")
    G = ox.add_edge_speeds(G)  # adds speed_kph if missing
    G = ox.add_edge_travel_times(G)  # adds travel_time from speed and length
    return G

# Map coordinates to nearest OSM node
def get_nearest_nodes(G, coords):
    lats, lons = zip(*coords)
    return ox.distance.nearest_nodes(G, X=lons, Y=lats)

# Build travel time matrix between all nodes (NxN)
def build_travel_time_matrix(G, node_ids):
    N = len(node_ids)
    time_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                time_matrix[i][j] = 0.0
            else:
                try:
                    path = nx.shortest_path(G, node_ids[i], node_ids[j], weight="travel_time")
                    time = nx.path_weight(G, path, weight="travel_time")
                    time_matrix[i][j] = time
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    time_matrix[i][j] = float('inf')
    return time_matrix
