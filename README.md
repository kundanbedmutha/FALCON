# MARL Cloud Resource Management

Multi-Agent Reinforcement Learning for intelligent cloud resource scheduling,
combining IQL, Fuzzy Logic preprocessing, and LLM-based dynamic reward shaping.

---

## Project Structure

```
marl_cloud_final/
├── train.py                  # Train IQL agents (with optional ablation mode)
├── benchmark.py              # Compare all 7 methods
├── prepare_results.py        # Convert results to plot-ready format
├── plot_results.py           # Generate all 6 paper figures
│
├── env/
│   └── cloud_env.py          # Cloud scheduling environment (20 machines, 105-dim obs)
│
├── agents/
│   ├── iql_agent.py          # Independent Q-Learning (4 cooperative agents)
│   ├── dqn_agent.py          # Single-agent DQN baseline
│   ├── ppo_agent.py          # Single-agent PPO baseline
│   └── baselines.py          # Round Robin and First Fit
│
├── fuzzy/
│   └── fuzzy_preprocessor.py # Fuzzy logic — adds 14 membership dims to obs
│
├── llm/
│   └── reward_shaper.py      # Dynamic reward weight switching (5 profiles)
│
└── results/                  # Auto-created — stores all JSON outputs
```

---

## Requirements

```
numpy
scipy
scikit-fuzzy
matplotlib
```

Install with:

```bash
pip install numpy scipy scikit-fuzzy matplotlib
```

No PyTorch required. All agents are implemented in pure NumPy.

---

## Quick Start

### Step 1 — Train the IQL agents

```bash
python train.py
```

Trains 4 cooperative IQL agents for 30 episodes with Fuzzy preprocessing and
LLM reward shaping. Saves results to `results/iql_training_history.json` and
`results/iql_agent_stats.json`.

### Step 2 — Run the full benchmark

```bash
python benchmark.py
```

Trains and evaluates all 7 methods:

1. IQL-MARL + Fuzzy + LLM (full system)
2. Single-Agent DQN
3. Single-Agent PPO
4. Round Robin
5. First Fit
6. H-MARL (Hierarchical)
7. Federated-MARL

Saves to `results/benchmark_results.json`.

**For quick testing** (shorter episodes, ~2 min total):

```bash
python benchmark.py --fast
```

### Step 3 — Generate plots

```bash
python prepare_results.py
python plot_results.py
```

Produces 6 publication-ready figures in `plots/`:

- `fig1_learning_curves.png`
- `fig2_benchmark_comparison.png`
- `fig3_utilisation.png`
- `fig4_sla_violations.png`
- `fig5_energy.png`
- `fig6_ablation.png`

### Step 4 — Run ablation study (optional)

```bash
python train.py --mode ablation
```

Trains and evaluates 3 variants (IQL Base, IQL+Fuzzy, IQL+Fuzzy+LLM) and
saves results to `results/ablation_results.json`.

---

## Key Components

### Environment (`env/cloud_env.py`)

- 20 machines, each with CPU / memory / network capacity
- Jobs arrive with random resource requirements, priorities, and deadlines
- Observation: 105-dim vector (machine states + current job features)
- Metrics tracked: SLA compliance, utilisation, latency, energy, violations

### IQL Agent (`agents/iql_agent.py`)

- 4 independent agents, each managing a subset of scheduling decisions
- 2-layer MLP Q-network (pure NumPy)
- Experience replay buffer (10,000 transitions)
- Gradient clipping at ±1.0 to prevent loss explosion
- Epsilon-greedy exploration with decay

### Fuzzy Preprocessor (`fuzzy/fuzzy_preprocessor.py`)

- Reads the first 8 dims of the observation (CPU util, memory util, load, job sizes)
- Outputs 14 fuzzy membership values (low/med/high for each variable)
- Final observation: 105 raw + 14 fuzzy = 119 dims

### LLM Reward Shaper (`llm/reward_shaper.py`)

Dynamically switches between 5 reward weight profiles based on live telemetry:

| Profile | Trigger | Focus |
|---|---|---|
| BALANCED | Default | Equal weights across all objectives |
| SLA_PRIORITY | SLA compliance < 45% | Heavy SLA penalty |
| ENERGY_SAVE | Avg energy > 2.8 | Reduce power consumption |
| THROUGHPUT | Utilisation < 60% | Maximise resource usage |
| LATENCY_FOCUS | Avg latency > 100ms | Minimise response time |

Profile re-evaluated every 200 steps.

### H-MARL (`benchmark.py` — method 6)

Two-level hierarchy:
- High-level DQN agent selects one of 4 machine clusters
- Low-level IQL agents schedule within the selected cluster
- Improves scalability over flat MARL for large deployments

### Federated MARL (`benchmark.py` — method 7)

- 3 independent nodes train on separate local data
- Every 5 episodes, Q-network weights are averaged across nodes (FedAvg)
- No raw data is shared — only model weights (privacy-preserving)

---

## Configuration

Edit the `CFG` dict at the top of `train.py` or `benchmark.py`:

```python
CFG = {
    "num_machines":   20,     # number of cloud machines
    "duration_steps": 2000,   # steps per episode
    "episodes":       30,     # training episodes
    "seed":           42,
    "lr":             1e-3,   # learning rate
    "gamma":          0.99,   # discount factor
    "batch_size":     64,
    "hidden":         256,    # hidden layer size
}
```

---

## Output Files

| File | Description |
|---|---|
| `results/iql_training_history.json` | Per-episode training metrics (reward, SLA, utilisation, latency, energy, LLM profile) |
| `results/iql_agent_stats.json` | Per-agent epsilon, buffer size, and average loss |
| `results/benchmark_results.json` | Eval metrics for all 7 methods |
| `results/ablation_results.json` | Ablation study results (created by `--mode ablation`) |
| `results/all_results.json` | Combined file used by `plot_results.py` |

---

## Common Issues

**SLA violations too high**
Increase training episodes and check that agent losses are not exploding.
Gradient clipping is already applied but you can also lower the learning rate.

**Benchmark is slow**
Use `python benchmark.py --fast` for quick iteration. The fast mode uses
500-step episodes instead of 2000.

**LLM profile stays on BALANCED**
This is normal for the first 200 steps. Profile switching activates after the
first evaluation window. Run more episodes to see profile changes logged.

**Missing `results/iql_training_history.json` when running `prepare_results.py`**
Run `python train.py` first to generate the training history file.
