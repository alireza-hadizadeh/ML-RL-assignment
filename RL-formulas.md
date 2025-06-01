### **\[Chapter 1: Introduction to RL]**

* **Reward Hypothesis**:

  > All goals can be described by the maximization of expected cumulative reward.

* **State Function**:

  > $S_t = f(H_t)$

* **Agent State**:

  > $S_a = f(H_t)$

---

### **\[Chapter 2: Bandits and Evaluative Feedback]**

* **Action Value Estimate (sample average)**:

  > $Q_t(a) = \frac{r_1 + r_2 + \ldots + r_k}{k_a}$

* **Incremental update rule**:

  > $Q_{k+1} = Q_k + \frac{1}{k+1}(r_k - Q_k)$

* **Nonstationary update rule**:

  > $Q_{k+1} = Q_k + \alpha(r_{k+1} - Q_k)$

* **Softmax (Gibbs/Boltzmann) action selection**:

  > $P(a) = \frac{e^{Q_t(a)/\tau}}{\sum_b e^{Q_t(b)/\tau}}$

* **Reinforcement Comparison**:

  > $p_{t+1}(a) = p_t(a) + \alpha(r_t - \bar{r}_t)$
  > $\bar{r}_{t+1} = \bar{r}_t + \alpha(r_t - \bar{r}_t)$

* **Pursuit method probability update**:

  > $\pi_{t+1}(a^*) = \pi_t(a^*) + \beta(1 - \pi_t(a^*))$

---

### **\[Chapter 3: RL Problem and Bellman Equations]**

* **Return (Episodic)**:

  > $R_t = r_{t+1} + r_{t+2} + \ldots + r_T$

* **Return (Discounted)**:

  > $R_t = \sum_{k=0}^\infty \gamma^k r_{t+k+1}$

* **Value Function**:

  > $V^\pi(s) = \mathbb{E}_\pi[R_t | S_t = s]$

* **Action-Value Function**:

  > $Q^\pi(s,a) = \mathbb{E}_\pi[R_t | S_t = s, A_t = a]$

* **Bellman Expectation Equation**:

  > $V^\pi(s) = \sum_a \pi(s,a) \sum_{s'} P_{ss'}^a [R_{ss'}^a + \gamma V^\pi(s')]$

* **Bellman Optimality Equation**:

  > $V^*(s) = \max_a \sum_{s'} P_{ss'}^a [R_{ss'}^a + \gamma V^*(s')]$
  > $Q^*(s,a) = \sum_{s'} P_{ss'}^a [R_{ss'}^a + \gamma \max_{a'} Q^*(s', a')]$

---

### **\[Chapter 4: Dynamic Programming]**

* **Policy Evaluation Update Rule**:

  > $V_{k+1}(s) = \sum_a \pi(s,a) \sum_{s'} P_{ss'}^a [R_{ss'}^a + \gamma V_k(s')]$

* **Policy Improvement Rule**:

  > $\pi'(s) = \arg\max_a Q^\pi(s,a)$

* **Value Iteration Update**:

  > $V_{k+1}(s) = \max_a \sum_{s'} P_{ss'}^a [R_{ss'}^a + \gamma V_k(s')]$

* **Generalized Policy Iteration**:
  Iterative interleaving of evaluation and improvement.

---

### **\[Chapter 5: Monte Carlo Methods]**

* **Monte Carlo Return**:

  > $R_t = r_{t+1} + r_{t+2} + \ldots + r_T$

* **First-Visit MC Update**:

  > $V(s) = \text{average of returns following first visits to } s$

* **Incremental MC Update**:

  > $V_{n+1} = V_n + \frac{1}{n}(R - V_n)$

* **Off-policy Weighted Return**:

  > $V = \frac{\sum_{k=1}^n w_k R_k}{\sum_{k=1}^n w_k}$

---

### **\[Chapter 6: Temporal Difference Learning]**

* **TD(0) Update Rule**:

  > $V(S_t) \leftarrow V(S_t) + \alpha[r_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$

* **SARSA Update (On-policy)**:

  > $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma Q(s',a') - Q(s,a)]$

* **Q-Learning Update (Off-policy)**:

  > $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$

* **R-Learning** (for average reward):

  > $Q(s,a) \leftarrow Q(s,a) + \alpha[r - \rho + \max_{a'} Q(s',a') - Q(s,a)]$
  > $\rho \leftarrow \rho + \beta[r - \rho + \max_{a'} Q(s',a') - \max_a Q(s,a)]$

---

### **\[Chapter 7: Eligibility Traces]**

* **n-step Return**:

  > $R^{(n)}_t = r_{t+1} + \gamma r_{t+2} + \ldots + \gamma^{n-1} r_{t+n} + \gamma^n V(s_{t+n})$

* **TD(λ) Forward View (λ-return)**:

  > $R^\lambda_t = (1 - \lambda) \sum_{n=1}^{T-t-1} \lambda^{n-1} R^{(n)}_t$

* **TD(λ) Backward View (Eligibility Trace)**:

  > $e_t(s) = \gamma \lambda e_{t-1}(s) + 1 \text{ if } s = s_t$
  > $V(s) \leftarrow V(s) + \alpha \delta_t e(s)$

* **SARSA(λ) Update**:

  > $Q(s,a) \leftarrow Q(s,a) + \alpha \delta_t e(s,a)$

---

### **\[Chapter 8: Cooperative Q-Learning]**

* **Q-learning update (Single Agent)**:

  > $Q(s,a) = r + \gamma \max_{a'} Q(s', a')$

* **Q-sharing (Simple Averaging)**:

  > $Q(s,a) = \frac{1}{n} \sum_{j=1}^n Q_j(s,a)$

* **Q-sharing with expertness**:

  > $Q_i = \sum_{j=1}^n W_{ij} Q_j$

* **Weighting by expertness**:

  > $W_{ij} = \frac{e_j}{\sum_k e_k}$

