# eval.py
import torch
from env import DTSPTDS
from model import DGTA
from tqdm import tqdm
import numpy as np

# ---------------- Settings ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "dgta_rl.pt"
NUM_INSTANCES = 100   # Number of test problems
GRAPH_SIZE = 20       # Same as training
TIME_HORIZON = 12     # Same as training

# ---------------- Load Model ----------------
model = DGTA().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------------- Evaluation Loop ----------------
rewards = []

with torch.no_grad():
    for _ in tqdm(range(NUM_INSTANCES), desc="Evaluating"):
        env = DTSPTDS(N=GRAPH_SIZE, T=TIME_HORIZON, device=DEVICE)
        state = env.reset()
        done = False

        while not done:
            coords = state['coords'].unsqueeze(0)             # (1, N, 2)
            t_idx = state['t_idx'].to(DEVICE)                 # (1,)
            visited = state['visited'].unsqueeze(0)           # (1, N)
            current = torch.tensor([[state['curr']]], device=DEVICE)  # (1, 1)

            logits = model(coords, t_idx, visited, current)
            action = torch.argmax(logits, dim=-1).item()      # Greedy choice
            state, reward, done, _ = env.step(action)

        rewards.append(reward)

# ---------------- Results ----------------
rewards = np.array(rewards)
print("\n--- Evaluation Summary ---")
print(f"Number of instances:     {NUM_INSTANCES}")
print(f"Average tour reward:     {rewards.mean():.2f}")
print(f"Best reward (min time):  {rewards.max():.2f}")
print(f"Worst reward:            {rewards.min():.2f}")
print(f"Std deviation:           {rewards.std():.2f}")
