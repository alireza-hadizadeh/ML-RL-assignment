# dgta_rl/train.py
import torch, random, math
from torch.optim import Adam
from scipy import stats
from tqdm import trange
from env import DTSPTDS
from dgta import DGTA
import os

# ------------------ hyper-parameters ------------------
BATCH   = 128
ITER    = 50                # paper uses 100 but 50 converges 
LR      = 1e-4
CLIP    = 1.0
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ helpers ------------------
def baseline_update_needed(loss_pi, loss_b, alpha=0.05):
    """Paired t‑test as in Algorithm 1  ."""
    t, p = stats.ttest_rel(loss_pi, loss_b)
    return p/2 < alpha and t < 0   # one‑sided (π better)

# ------------------ main loop ------------------
def train():
    env = DTSPTDS(N=20, T=12, device=DEVICE)
    model = DGTA().to(DEVICE)
    base  = DGTA().to(DEVICE)      # baseline copy
    base.load_state_dict(model.state_dict())

    opt = Adam(model.parameters(), lr=LR)
    print(f"Using device: {DEVICE}")
    log_file = open("dgta_rl.log", "w")
    log_file.write("iter,avg_reward_pi,avg_reward_baseline\n")
    for it in trange(ITER, desc="DGTA‑RL training"):
        batch_loss_pi, batch_loss_b = [], []

        for _ in range(BATCH):
            # --- generate one trajectory with sampling ---
            state = env.reset()
            logps, rewards = [], []
            done = False
            while not done:
                logits = model(state['coords'].unsqueeze(0),
                               state['t_idx'],
                               state['visited'].unsqueeze(0),
                               torch.tensor([[state['coords'].size(0)]],
                                            device=DEVICE))
                prob = torch.softmax(logits, dim=-1).squeeze(0)
                action = torch.multinomial(prob, 1).item()
                logps.append(torch.log(prob[action]))
                state, reward, done, _ = env.step(action)
            batch_loss_pi.append(-reward)
            loss = sum(logps) * reward     # REINFORCE

            # --- baseline greedy rollout ---
            state_b = env.reset()
            done = False
            while not done:
                logits = base(state_b['coords'].unsqueeze(0),
                              state_b['t_idx'],
                              state_b['visited'].unsqueeze(0),
                              torch.tensor([[state_b['coords'].size(0)]],
                                           device=DEVICE))
                action = torch.argmax(logits, dim=-1).item()
                state_b, reward_b, done, _ = env.step(action)
            batch_loss_b.append(-reward_b)

            # optimise
            opt.zero_grad()
            (-loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            # Save every 500 iterations
            if (it + 1) % 500 == 0:
                torch.save(model.state_dict(), f"dgta_rl_iter_{it+1}.pt")

        # ----- paired t‑test -----
        if baseline_update_needed(batch_loss_pi, batch_loss_b):
            base.load_state_dict(model.state_dict())

    avg_pi = sum(batch_loss_pi) / len(batch_loss_pi)
    avg_b  = sum(batch_loss_b) / len(batch_loss_b)
    log_file.write(f"{it+1},{-avg_pi:.4f},{-avg_b:.4f}\n")
    log_file.flush()  # in case of crash

    torch.save(model.state_dict(), "dgta_rl.pt")
    
    log_file.close()

if __name__ == "__main__":
    train()
