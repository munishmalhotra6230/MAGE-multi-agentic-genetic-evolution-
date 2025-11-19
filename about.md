

---

# **MAGE: A Multi-Agent Genetic Evolution Framework for Neural Architecture Optimization**

---

## **Abstract**

The **MAGE (Multi-Agent Genetic Evolution)** algorithm is a neural network optimization framework that merges evolutionary computation with adaptive randomness to evolve architectures, weights, and hyperparameters simultaneously.
Initially designed as a lightweight self-evolving trainer, MAGE now integrates two distinct search paradigms — *Single-Architecture Evolution* and *Multi-Architecture Exploration* — allowing flexible behavior depending on research goals.
Unlike classical Neuroevolution or NAS methods that rely on complex graph search, MAGE operates on compact genetic representations of neural networks, promoting *diversity through randomness* and *stability through fitness control*.
This paper presents the conceptual foundation, system architecture, mathematical formulation, pseudocode, and implementation flow of MAGE.

---

## **1. Introduction**

Neural network optimization traditionally depends on gradient descent methods tuned by human-engineered architectures and learning rates. However, such manual tuning limits scalability and adaptability.

**MAGE** was conceived to automate this process using a population-based evolutionary framework where each *agent* represents an independent neural model with its own parameters, architecture, and hyperparameters. These agents evolve collectively — learning locally, competing globally, and mutating based on performance.

The initial design of MAGE focused purely on *randomness-based self-organization*, inspired by population diversity in natural systems. The randomness was the engine of discovery — agents began with random weights, random architectures, and minimal rules. However, this produced instability at scale.

The modern form of MAGE introduces **controlled randomness**, **adaptive mutation**, and **search modes** (single and multi) to ensure both stability and exploratory capacity. This paper formalizes that final design.

---

## **2. Evolution of the MAGE Concept**

| Generation | Core Idea                                                | Limitation                           | Resulting Evolution                          |
| ---------- | -------------------------------------------------------- | ------------------------------------ | -------------------------------------------- |
| MAGE v1    | Random weight diversification per agent                  | High variance, unstable training     | Introduced noise control                     |
| MAGE v2    | Added mutation control for hyperparams                   | Lack of structure search             | Introduced architecture mutation             |
| MAGE v3    | Added search modes: *single* and *multi*                 | Needed deterministic reproducibility | Added controlled randomness (seed-based RNG) |
| Final MAGE | Unified architecture + hyperparameter + weight evolution | None (stable and research-ready)     | Current class design                         |

Thus, MAGE evolved from a random experimentation framework to a structured evolutionary system combining **determinism, exploration, and self-adaptation**.

---

## **3. Core Architecture of the MAGE Class**

### **3.1 Class Overview**

Each instance of `MAGE` represents an *evolutionary trainer* with:

* **Population of agents:** Each agent is a neural network defined by architecture, weights, and hyperparameters.
* **Fitness function:** Validation loss (val_loss) determines evolutionary fitness.
* **Evolutionary loop:** Repeated selection, mutation, and replacement across master epochs.

### **3.2 Core Components**

| Component                                                              | Description                                                            |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `__init__()`                                                           | Initializes search settings, noise factors, and RNG.                   |
| `fit()`                                                                | Performs evolution loop combining local learning and global selection. |
| `_train_agent_local()`                                                 | Trains a single agent locally using standard backpropagation.          |
| `_fitness()`                                                           | Computes validation loss + complexity penalty.                         |
| `_mutate_weights()`, `_mutate_hyperparams()`, `_mutate_architecture()` | Apply stochastic transformations to agents.                            |
| `get_execution_details()`                                              | Logs evolution history for research analysis.                          |
| `evaluate()`                                                           | Evaluates best-evolved agent on test data.                             |

---

## **4. Algorithm Workflow**

### **4.1 Population Initialization**

Each agent begins as:
[
A_i = {W_i, B_i, arch_i, hp_i}
]
where:

* (W_i, B_i) = weight and bias matrices
* (arch_i = {layers, activations})
* (hp_i = {lr, batch, optimizer, weight_decay})

Two modes exist:

* **Single Mode:** All agents share *same architecture*, different weights/hyperparameters.
* **Multi Mode:** Each agent starts with *different architecture* from `search_space`.

---

### **4.2 Local Learning (Within Master Epoch)**

Each agent trains for a few `local_epochs` using its own hyperparameters.

Pseudocode:

```
for agent in population:
    agent.train(X_train, y_train, epochs=local_epochs)
    agent.val_loss = evaluate(agent, X_val, y_val)
```

This produces per-agent `val_loss` values used for selection.

---

### **4.3 Global Selection & Fitness Computation**

Fitness is inversely proportional to performance:
[
F_i = ValLoss_i + \lambda \cdot Complexity(arch_i)
]
Lower fitness is better.

Agents are sorted by fitness; elites (top N) are preserved for next generation.

---

### **4.4 Mutation and Controlled Randomness**

For each non-elite agent:

1. Select parent with better fitness.
2. Create child copy or random agent (depending on search mode).
3. Apply probabilistic mutations:

[
p_{mutate} = base_prob \times rank_factor
]

The **rank factor** (between 1.0 and 0.3) ensures *stronger agents mutate less, weaker agents mutate more*, maintaining diversity.

---

### **4.5 Controlled Randomness (Deterministic RNG)**

MAGE introduces deterministic randomness by:

```python
self.rng = np.random.default_rng(seed)
```

and uses it consistently across weight initialization, architecture mutation, and hyperparameter evolution.

This ensures repeatability while maintaining diversity.

---

## **5. Search Modes**

### **5.1 Single Mode**

* Architecture: fixed by user.
* Variation: only through weights and hyperparameters.
* Purpose: evaluate optimization strategies without structural search.
* Example:

  ```python
  search_mode="single"
  search_space={"layers": [[784, 128, 10]], "activations": ["relu"]}
  ```

Lifecycle:

1. All agents start with same `[784,128,10]`.
2. Each agent trains independently.
3. Best weights evolve through mutation and selection.

### **5.2 Multi Mode**

* Architecture: sampled from `search_space`.
* Each agent evolves both structure and parameters.
* Used for *Neural Architecture Search (NAS)*-like behavior.
* Example:

  ```python
  search_mode="multi"
  search_space={
      "layers": [[64,128,64],[128,64],[256,128,64],[512,256]],
      "activations": ["relu","tanh","gelu"]
  }
  ```

Lifecycle:

1. Different architectures initialized per agent.
2. Training + evaluation performed per structure.
3. Best-performing architectures and hyperparameters survive.
4. Mutations can add/remove layers, neurons, or activations.

---

## **6. Evolutionary Loop Summary**

Pseudocode Summary:

```
Initialize population (single or multi)
for each master_epoch:
    Train each agent locally
    Evaluate validation loss (val_loss)
    Select top elite agents
    Mutate and reproduce remaining agents
    Update controlled randomness
Save best architecture and weights
```

---

## **7. Validation Loss Lifecycle**

* Computed after each local training loop:
  [
  val_loss = L(f(X_{val};W,B), y_{val})
  ]
* Stored in agent dict and used to compute fitness.
* Drives selection pressure — agents with lower val_loss survive.
* In `get_execution_details()`, logged per master epoch for analysis.

In **single mode**, val_loss tracks how different hyperparameters impact same architecture.
In **multi mode**, val_loss additionally measures *architectural fitness*.

---

## **8. Experimental Observations**

| Mode   | Objective                    | Expected Behavior                                         |
| ------ | ---------------------------- | --------------------------------------------------------- |
| Single | Optimize one given structure | Finds best weights/lr faster, stable val_loss convergence |
| Multi  | Explore architectures        | Slower convergence but more global minima exploration     |
| Both   | Complementary                | Combine to fine-tune architecture post exploration        |

---

## **9. Key Features Making MAGE Research-Ready**

1. **Deterministic RNG:** Reproducible randomness.
2. **Controlled Mutation Scaling:** Based on agent rank.
3. **Dual-Mode Search:** Single (fixed) vs Multi (exploratory).
4. **CSV Logging:** Enables full traceability of each agent’s evolution.
5. **Architecture & Hyperparam Integration:** Unified evolutionary process.
6. **Complexity-Aware Fitness:** Penalizes overly large models.
7. **Self-contained Python Implementation:** No dependency on heavy NAS frameworks.

---

## **10. Example Result Interpretation (CSV Log)**

Example Row:

```
3,0,0.265,101770,"[784,128,10]",['relu'],0.0009,128
```

Meaning:

* Epoch 3, Agent 0
* Validation Loss = 0.265
* Parameter Count = 101,770
* Architecture = [784,128,10]
* Activation = relu
* Learning Rate = 0.0009
* Batch Size = 128

This row represents one agent’s final status after training and selection for a given master epoch.

---

## **11. Conclusion**

MAGE represents a minimal yet complete **evolutionary intelligence system** that autonomously explores, evaluates, and optimizes neural architectures and hyperparameters. Its design balances *stochastic exploration* with *deterministic reproducibility*, making it suitable for research on evolutionary deep learning, NAS, and adaptive optimization.

The dual-mode framework enables both **deep optimization (single)** and **broad exploration (multi)** — making MAGE a scalable foundation for future agentic AI research.

---

## **12. Future Directions**

* Integrate **self-adaptive learning rates** per agent.
* Add **crossover mechanism** for hybrid offspring generation.
* Implement **parallel execution** across CPU/GPU nodes.
* Extend **fitness** to multi-objective (accuracy, latency, energy).

---

## **Appendix: Pseudocode of the MAGE Algorithm**

```
Initialize population of N agents
for master_epoch in range(M):
    for each agent:
        train locally for local_epochs
        compute val_loss
    rank agents by fitness (val_loss + complexity)
    select top elites
    for each non-elite agent:
        if mode == "multi":
            mutate architecture
        mutate weights and hyperparams
        apply controlled randomness
    replace old population
return best agent
```

---



