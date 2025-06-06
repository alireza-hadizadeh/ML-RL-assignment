# DGTA-RL: Dynamic Graph Temporal Attention Reinforcement Learning

This repository contains a PyTorch implementation of the DGTA-RL model, designed to solve the **Dynamic Traveling Salesman Problem with Time-Dependent and Stochastic Travel Times (DTSP-TDS)**.

## 📦 Project Structure

```
DGTA-RL/
├── model/              # Modular model components
│   ├── dgta.py
│   ├── dual_attention.py
│   ├── dynamic_encoder.py
│   ├── pointer_decoder.py
│   ├── positional.py
├── env.py              # DTSP-TDS environment simulator
├── train.py            # REINFORCE training loop with logging
├── eval.py             # Evaluation on test instances
├── plot.py             # Visualizes reward trends
├── utils.py            # Helper functions
└── dgta_rl.log         # Training reward logs (generated after training)
```

## 🚀 Getting Started

### 1. Clone and Install

```bash
git clone https://github.com/alireza-hadizadeh/ML-RL-assignment.git
cd ML-RL-assignment/DGTA-RL
pip install -r requirements.txt  # (Coming soon)
```

### 2. Train the Model

```bash
python -m train
```

This trains the DGTA model on synthetic DTSP-TDS instances.

### 3. Evaluate the Trained Model

```bash
python eval.py
```

Outputs average, best, and worst tour quality over 100 test instances.

### 4. Plot Reward Progress

```bash
python plot.py
```

Saves a PNG showing training reward curves for policy vs baseline.

## 🧠 Model Highlights

* **Graph Transformer with Dual Attention (DGTA)**
* Learns from scratch using **REINFORCE**
* **Gamma-distributed** stochastic travel times
* **Paired t-test** policy update (baseline-aware)

## 📝 Notes

* Trained models are saved as `dgta_rl.pt`
* Intermediate checkpoints saved as `dgta_rl_iter_XXXX.pt`
* Logging format: `dgta_rl.log`

## 📚 Reference

Based on the 2025 article:

> Chen et al., "The Dynamic Traveling Salesman Problem with Time-Dependent and Stochastic Travel Times: A Deep Reinforcement Learning Approach"

---

For questions or improvements, feel free to open issues or pull requests! 💡
