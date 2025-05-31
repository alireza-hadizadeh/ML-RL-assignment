# plot.py
import matplotlib.pyplot as plt
import pandas as pd

# ---------------- Load log file ----------------
log_path = "dgta_rl.log"

try:
    df = pd.read_csv(log_path)

    # ---------------- Plot ----------------
    plt.figure(figsize=(10, 6))
    plt.plot(df['iter'], df['avg_reward_pi'], label='Policy Reward (REINFORCE)', linewidth=2)
    plt.plot(df['iter'], df['avg_reward_baseline'], label='Baseline Reward (Greedy)', linewidth=2, linestyle='--')

    plt.title("DGTA-RL Training Reward Over Iterations", fontsize=14)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Total Tour Time (Lower is Better)", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("dgta_rl_training_plot.png", dpi=150)
    plt.show()

except FileNotFoundError:
    print(f"[ERROR] Log file not found: {log_path}")
