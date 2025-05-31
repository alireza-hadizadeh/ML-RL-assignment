# dgta_rl/env.py
import numpy as np
from scipy.stats import gamma
import torch

class DTSPTDS:
    """
    A lightweight simulator for the Dynamic TSP with
    time‑dependent & stochastic travel times (DTSP‑TDS).

    Nodes are numbered 0..N‑1 with node 0 == depot.
    """
    def __init__(self, N=20, T=12, sigma=15, beta=1.0,
                 horizon=240, seed=None, device="cpu"):
        self.N, self.T, self.device = N, T, device
        self.delta_t = horizon // T        # Δt length (minutes)
        rng = np.random.default_rng(seed)

        # Random 2‑D coordinates in a 100×100 square
        self.coords = rng.uniform(0, 100, size=(N, 2))

        # Pre‑compute expected speed matrix û_ij(t) (Eq. in section 5.1) 
        self.u_max = np.array([26, 36, 50])     # zone max speeds
        self.b = np.array([0.5, 1.0, 0.5])      # congestion level
        self.u_hat = self._expected_speeds()

        self.sigma, self.beta = sigma, beta     # γ‑distribution params
        self.state = None
        self.reset()

    # ------------------------------------------------------------------
    def _expected_speeds(self):
        # Zone assignment by concentric rings (r=0–20,20–40,>40)  
        depot = self.coords[0]
        dists = np.linalg.norm(self.coords - depot, axis=1)
        zones = np.digitize(dists, (20, 40))
        N = self.N
        speed = np.zeros((self.T, N, N))
        for t in range(self.T):
            h = t // (self.T // 3)              # congestion period index
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    z = zones[i]                # departing zone
                    speed[t, i, j] = self.b[h] * self.u_max[z]
        return speed                            # shape: (T, N, N)

    # ------------------------------------------------------------------
    def _travel_time(self, i, j, t_dep):
        """Gamma‑distributed travel time realisation  ."""
        if i == j:
            return 0.
        mean_speed = self.u_hat[t_dep, i, j]
        dist = np.linalg.norm(self.coords[i] - self.coords[j])
        expected_tt = dist / mean_speed
        shape = (expected_tt / self.beta)
        return gamma(a=shape, scale=self.beta).rvs()

    # ------------------------------------------------------------------
    def reset(self):
        self.visited = np.zeros(self.N, dtype=bool)
        self.visited[0] = True                  # depot visited
        self.t = 0                              # time index (0..T‑1)
        self.curr = 0                           # current node idx
        self.tour = [0]
        self.elapsed = 0.0
        self.state = self._build_state()
        return self.state

    # ------------------------------------------------------------------
    def _build_state(self):
        # State = (coords, time idx, visited mask, elapsed time, current node)
        return dict(
            coords=torch.tensor(self.coords, dtype=torch.float32, device=self.device),
            t_idx=torch.tensor([self.t], dtype=torch.long, device=self.device),
            visited=torch.tensor(self.visited, dtype=torch.bool, device=self.device),
            elapsed=torch.tensor([self.elapsed], dtype=torch.float32, device=self.device),
            curr=torch.tensor(self.curr, dtype=torch.long, device=self.device)  # ← Add this line
        )


    # ------------------------------------------------------------------
    def step(self, action):
        """Action = next node to visit."""
        i, j = self.curr, int(action)
        if self.visited[j]:
            raise ValueError("Visited node chosen again.")
        # Compute realised travel time
        tt = self._travel_time(i, j, self.t)
        self.elapsed += tt
        self.t = int((self.elapsed // self.delta_t) % self.T)
        self.curr = j
        self.visited[j] = True
        self.tour.append(j)

        done = self.visited.all()
        reward = -self.elapsed if done else 0.0
        return self._build_state(), reward, done, {}

    # ------------------------------------------------------------------
    def legal_actions(self):
        return np.flatnonzero(~self.visited)
