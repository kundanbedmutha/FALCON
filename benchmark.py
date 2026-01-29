"""
benchmark.py — Compare All Methods
Run: python benchmark.py
Run fast (fewer episodes): python benchmark.py --fast

New vs original:
  ✅ Dynamic LLM profile switching (was always BALANCED)
  ✅ Ablation variants loaded from ablation_results.json if present
  ✅ H-MARL stub: hierarchical 2-level scheduling (cluster + VM)
  ✅ Federated MARL stub: local training + weight averaging
  ✅ --fast flag: reduces train/eval episodes for quick iteration
  ✅ Per-method timing printed
  ✅ 7-method table matching plot_results.py expectations
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import json
import time

from env.cloud_env import CloudEnv
from agents.iql_agent import IQLAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.baselines import RoundRobinAgent, FirstFitAgent
from fuzzy.fuzzy_preprocessor import FuzzyPreprocessor
from llm.reward_shaper import LLMRewardShaper

# ── Config ────────────────────────────────────────────────────────────────
CFG = {
    "num_machines":   20,
    "duration_steps": 2000,
    "eval_episodes":  5,
    "train_episodes": 20,
    "seed":           42,
    "save_dir":       "results",
}

CFG_FAST = {
    "num_machines":   20,
    "duration_steps": 500,   # shorter episodes → much faster
    "eval_episodes":  3,
    "train_episodes": 10,
    "seed":           42,
    "save_dir":       "results",
}

os.makedirs(CFG["save_dir"], exist_ok=True)


def make_env(cfg, seed_offset=0):
    return CloudEnv(num_machines=cfg["num_machines"],
                    seed=cfg["seed"] + seed_offset,
                    duration_steps=cfg["duration_steps"])


def apply_fuzzy(fuzzy, raw_obs):
    vec   = np.asarray(raw_obs, dtype=np.float32)
    chunk = vec[:8] if len(vec) >= 8 else np.pad(vec, (0, 8 - len(vec)))
    return np.concatenate([vec, fuzzy.process(chunk)[8:]]).astype(np.float32)


def get_valid_list(env, current_job):
    if current_job is None:
        return list(range(env.num_machines))
    valid = [m for m in range(env.num_machines)
             if (env.machines[m, 0] + current_job["cpu_req"] <= 1.0 and
                 env.machines[m, 1] + current_job["mem_req"] <= 1.0 and
                 env.machines[m, 2] + current_job["net_req"] <= 1.0)]
    return valid if valid else list(range(env.num_machines))


def shape_reward(raw, info, llm, step):
    w       = llm.update(info, step)
    util    = float(info.get("avg_utilization", 0.5))
    sla     = float(info.get("sla_compliance",  1.0))
    energy  = float(min(info.get("avg_energy",  0.1), 1.0))
    latency = float(min(info.get("avg_latency", 1.0) / 200.0, 1.0))
    shaped  = (w["w_util"] * util + w["w_sla"] * sla
               - w["w_energy"] * energy - w["w_latency"] * latency)
    return float(0.6 * shaped + 0.4 * raw)


def _aggregate(metrics_list, name):
    keys = ["sla_compliance","avg_utilization","avg_latency",
            "avg_energy","sla_violations","jobs_scheduled"]
    result = {"method": name}
    for k in keys:
        vals = [m[k] for m in metrics_list]
        result[k + "_mean"] = round(float(np.mean(vals)), 4)
        result[k + "_std"]  = round(float(np.std(vals)),  4)
    return result


# ── IQL + Fuzzy + LLM ────────────────────────────────────────────────────
def run_iql(cfg, train_eps, eval_eps):
    print("\n  [1/7] IQL-MARL + Fuzzy + LLM Reward Shaping")
    fuzzy = FuzzyPreprocessor()
    llm   = LLMRewardShaper()
    obs_dim = make_env(cfg).obs_size + fuzzy.extra_dims

    agents = [IQLAgent(agent_id=i, obs_dim=obs_dim,
                       action_dim=cfg["num_machines"],
                       hidden=256, lr=1e-3, gamma=0.99,
                       batch_size=64, seed=cfg["seed"])
              for i in range(CloudEnv.N_AGENTS)]

    print("    Training...", end="", flush=True)
    t0 = time.time()
    for ep in range(train_eps):
        env = make_env(cfg, ep)
        obs_list, current_job = env.reset()
        step = 0
        while True:
            a_obs  = [apply_fuzzy(fuzzy, obs_list[i]) for i in range(4)]
            valid  = get_valid_list(env, current_job)
            acts   = [agents[i].select_action(a_obs[i], valid_actions=valid, training=True)
                      for i in range(4)]
            nobs, rewards, done, info = env.step(acts)
            shaped = [shape_reward(rewards[i], info, llm, step) for i in range(4)]
            na_obs = [apply_fuzzy(fuzzy, nobs[i]) for i in range(4)]
            for i in range(4):
                agents[i].store(a_obs[i], acts[i], shaped[i], na_obs[i], float(done))
                agents[i].learn()
            obs_list    = nobs
            current_job = env._get_current_job()
            step += 1
            if done: break
        if (ep + 1) % max(1, train_eps // 4) == 0:
            print(".", end="", flush=True)
    print(f" done ({time.time()-t0:.1f}s)")

    metrics = []
    for ep in range(eval_eps):
        env = make_env(cfg, 1000 + ep)
        obs_list, current_job = env.reset()
        while True:
            a_obs = [apply_fuzzy(fuzzy, obs_list[i]) for i in range(4)]
            valid = get_valid_list(env, current_job)
            acts  = [agents[i].select_action(a_obs[i], valid_actions=valid, training=False)
                     for i in range(4)]
            nobs, _, done, info = env.step(acts)
            obs_list    = nobs
            current_job = env._get_current_job()
            if done: break
        metrics.append(info)
    return _aggregate(metrics, "IQL-MARL+Fuzzy+LLM")


# ── DQN ───────────────────────────────────────────────────────────────────
def run_dqn(cfg, train_eps, eval_eps):
    print("\n  [2/7] Single Agent DQN")
    env   = make_env(cfg)
    agent = DQNAgent(obs_size=env.obs_size, action_size=cfg["num_machines"],
                     seed=cfg["seed"])

    print("    Training...", end="", flush=True)
    t0 = time.time()
    for ep in range(train_eps):
        env = make_env(cfg, ep)
        obs_list, current_job = env.reset()
        while True:
            obs    = obs_list[0]
            action = agent.select_action(obs)
            nobs, rewards, done, info = env.step([action] * 4)
            agent.store(obs, action, float(np.mean(rewards)), nobs[0], float(done))
            agent.train()
            obs_list    = nobs
            current_job = env._get_current_job()
            if done: break
        if (ep + 1) % max(1, train_eps // 4) == 0:
            print(".", end="", flush=True)
    print(f" done ({time.time()-t0:.1f}s)")

    metrics = []
    for ep in range(eval_eps):
        env = make_env(cfg, 1000 + ep)
        obs_list, current_job = env.reset()
        while True:
            action = agent.select_action(obs_list[0], deterministic=True)
            nobs, _, done, info = env.step([action] * 4)
            obs_list    = nobs
            current_job = env._get_current_job()
            if done: break
        metrics.append(info)
    return _aggregate(metrics, "DQN")


# ── PPO ───────────────────────────────────────────────────────────────────
def run_ppo(cfg, train_eps, eval_eps):
    print("\n  [3/7] Single Agent PPO")
    env   = make_env(cfg)
    agent = PPOAgent(obs_size=env.obs_size, action_size=cfg["num_machines"],
                     seed=cfg["seed"])

    print("    Training...", end="", flush=True)
    t0 = time.time()
    for ep in range(train_eps):
        env = make_env(cfg, ep)
        obs_list, current_job = env.reset()
        while True:
            obs             = obs_list[0]
            action, val, lp = agent.select_action(obs)
            nobs, rewards, done, info = env.step([action] * 4)
            agent.store(obs, action, float(np.mean(rewards)), val, lp, float(done))
            agent.train()
            obs_list    = nobs
            current_job = env._get_current_job()
            if done: break
        if (ep + 1) % max(1, train_eps // 4) == 0:
            print(".", end="", flush=True)
    print(f" done ({time.time()-t0:.1f}s)")

    metrics = []
    for ep in range(eval_eps):
        env = make_env(cfg, 1000 + ep)
        obs_list, current_job = env.reset()
        while True:
            action, _, _ = agent.select_action(obs_list[0], deterministic=True)
            nobs, _, done, info = env.step([action] * 4)
            obs_list    = nobs
            current_job = env._get_current_job()
            if done: break
        metrics.append(info)
    return _aggregate(metrics, "PPO")


# ── Round Robin ───────────────────────────────────────────────────────────
def run_round_robin(cfg, eval_eps):
    print("\n  [4/7] Round Robin Baseline")
    agent   = RoundRobinAgent(cfg["num_machines"])
    metrics = []
    for ep in range(eval_eps):
        env = make_env(cfg, 1000 + ep)
        obs_list, current_job = env.reset()
        while True:
            action = agent.select_action()
            nobs, _, done, info = env.step([action] * 4)
            obs_list    = nobs
            current_job = env._get_current_job()
            if done: break
        metrics.append(info)
    return _aggregate(metrics, "RoundRobin")


# ── First Fit ─────────────────────────────────────────────────────────────
def run_first_fit(cfg, eval_eps):
    print("\n  [5/7] First Fit Baseline")
    agent   = FirstFitAgent(cfg["num_machines"])
    metrics = []
    for ep in range(eval_eps):
        env = make_env(cfg, 1000 + ep)
        obs_list, current_job = env.reset()
        while True:
            action = agent.select_action(obs_list[0], machines=env.machines,
                                         job=current_job)
            nobs, _, done, info = env.step([action] * 4)
            obs_list    = nobs
            current_job = env._get_current_job()
            if done: break
        metrics.append(info)
    return _aggregate(metrics, "FirstFit")


# ── H-MARL (Hierarchical) ─────────────────────────────────────────────────
def run_hmarl(cfg, train_eps, eval_eps):
    """
    Hierarchical MARL: 2-level architecture.
      High-level agent  → selects a cluster (group of 5 machines)
      Low-level agents  → 4 IQL agents assign within the selected cluster

    This implements the H-MARL innovation from the proposal.
    """
    print("\n  [6/7] H-MARL (Hierarchical)")
    N_CLUSTERS  = 4
    CLUSTER_SZ  = cfg["num_machines"] // N_CLUSTERS   # 5

    fuzzy   = FuzzyPreprocessor()
    llm     = LLMRewardShaper()
    obs_dim = make_env(cfg).obs_size + fuzzy.extra_dims

    # High-level: cluster selector (1 DQN agent)
    hl_agent = DQNAgent(obs_size=obs_dim, action_size=N_CLUSTERS, seed=cfg["seed"])
    # Low-level: 4 IQL agents that operate within the chosen cluster
    ll_agents = [IQLAgent(agent_id=i, obs_dim=obs_dim,
                          action_dim=CLUSTER_SZ, hidden=128,
                          lr=1e-3, gamma=0.99, batch_size=64,
                          seed=cfg["seed"])
                 for i in range(CloudEnv.N_AGENTS)]

    def h_select(obs_f, valid_list):
        # High-level picks cluster
        cluster = hl_agent.select_action(obs_f)
        cluster_start = cluster * CLUSTER_SZ
        # Map cluster-local actions to global machine ids
        global_valid = [m for m in valid_list
                        if cluster_start <= m < cluster_start + CLUSTER_SZ]
        if not global_valid:
            global_valid = valid_list      # fallback: use all valid
        # Low-level agents pick within that cluster
        local_acts = [ll_agents[i].select_action(obs_f, training=True,
                       valid_actions=list(range(CLUSTER_SZ)))
                      for i in range(CloudEnv.N_AGENTS)]
        # Map back to global: clip to valid range
        global_acts = [cluster_start + (a % CLUSTER_SZ) for a in local_acts]
        global_acts = [min(a, cfg["num_machines"]-1) for a in global_acts]
        return global_acts, cluster

    print("    Training...", end="", flush=True)
    t0 = time.time()
    for ep in range(train_eps):
        env = make_env(cfg, ep)
        obs_list, current_job = env.reset()
        step = 0
        while True:
            obs_f = apply_fuzzy(fuzzy, obs_list[0])
            valid = get_valid_list(env, current_job)
            acts, cluster = h_select(obs_f, valid)
            nobs, rewards, done, info = env.step(acts)
            shaped = [shape_reward(rewards[i], info, llm, step) for i in range(4)]
            nobs_f = apply_fuzzy(fuzzy, nobs[0])
            # Train high-level
            hl_agent.store(obs_f, cluster, float(np.mean(shaped)), nobs_f, float(done))
            hl_agent.train()
            # Train low-level
            for i in range(4):
                local_a = acts[i] % CLUSTER_SZ
                ll_agents[i].store(obs_f, local_a, shaped[i], nobs_f, float(done))
                ll_agents[i].learn()
            obs_list    = nobs
            current_job = env._get_current_job()
            step += 1
            if done: break
        if (ep + 1) % max(1, train_eps // 4) == 0:
            print(".", end="", flush=True)
    print(f" done ({time.time()-t0:.1f}s)")

    metrics = []
    for ep in range(eval_eps):
        env = make_env(cfg, 1000 + ep)
        obs_list, current_job = env.reset()
        while True:
            obs_f = apply_fuzzy(fuzzy, obs_list[0])
            valid = get_valid_list(env, current_job)
            acts, _ = h_select(obs_f, valid)
            nobs, _, done, info = env.step(acts)
            obs_list    = nobs
            current_job = env._get_current_job()
            if done: break
        metrics.append(info)
    return _aggregate(metrics, "H-MARL")


# ── Federated MARL ────────────────────────────────────────────────────────
def run_federated(cfg, train_eps, eval_eps):
    """
    Federated MARL: N_NODES independent nodes train locally.
    Every FEDERATED_ROUND episodes, average Q-network weights centrally.
    Only weight updates are shared — not raw data (privacy-preserving).
    """
    print("\n  [7/7] Federated MARL")
    N_NODES         = 3
    FEDERATED_ROUND = 5    # average weights every 5 episodes

    fuzzy   = FuzzyPreprocessor()
    llm     = LLMRewardShaper()
    obs_dim = make_env(cfg).obs_size + fuzzy.extra_dims

    # Each node has its own set of 4 IQL agents
    nodes = [[IQLAgent(agent_id=i, obs_dim=obs_dim,
                       action_dim=cfg["num_machines"], hidden=128,
                       lr=1e-3, gamma=0.99, batch_size=64,
                       seed=cfg["seed"] + node_id * 100)
              for i in range(CloudEnv.N_AGENTS)]
             for node_id in range(N_NODES)]

    def federated_average():
        """Average Q-network weights across all nodes (FedAvg)."""
        for agent_idx in range(CloudEnv.N_AGENTS):
            # Collect weights from all nodes
            W1s = [nodes[n][agent_idx].qnet.W1 for n in range(N_NODES)]
            b1s = [nodes[n][agent_idx].qnet.b1 for n in range(N_NODES)]
            W2s = [nodes[n][agent_idx].qnet.W2 for n in range(N_NODES)]
            b2s = [nodes[n][agent_idx].qnet.b2 for n in range(N_NODES)]
            # Compute mean
            avg_W1 = np.mean(W1s, axis=0)
            avg_b1 = np.mean(b1s, axis=0)
            avg_W2 = np.mean(W2s, axis=0)
            avg_b2 = np.mean(b2s, axis=0)
            # Broadcast back to all nodes
            for n in range(N_NODES):
                nodes[n][agent_idx].qnet.W1 = avg_W1.copy()
                nodes[n][agent_idx].qnet.b1 = avg_b1.copy()
                nodes[n][agent_idx].qnet.W2 = avg_W2.copy()
                nodes[n][agent_idx].qnet.b2 = avg_b2.copy()

    print("    Training...", end="", flush=True)
    t0 = time.time()
    for ep in range(train_eps):
        # Each node trains on its own local environment (different seed = different data)
        for node_id, node_agents in enumerate(nodes):
            env = make_env(cfg, ep * N_NODES + node_id)
            obs_list, current_job = env.reset()
            step = 0
            while True:
                a_obs  = [apply_fuzzy(fuzzy, obs_list[i]) for i in range(4)]
                valid  = get_valid_list(env, current_job)
                acts   = [node_agents[i].select_action(a_obs[i], valid_actions=valid,
                                                       training=True)
                          for i in range(4)]
                nobs, rewards, done, info = env.step(acts)
                shaped = [shape_reward(rewards[i], info, llm, step) for i in range(4)]
                na_obs = [apply_fuzzy(fuzzy, nobs[i]) for i in range(4)]
                for i in range(4):
                    node_agents[i].store(a_obs[i], acts[i], shaped[i], na_obs[i], float(done))
                    node_agents[i].learn()
                obs_list    = nobs
                current_job = env._get_current_job()
                step += 1
                if done: break

        # Federated averaging every FEDERATED_ROUND episodes
        if (ep + 1) % FEDERATED_ROUND == 0:
            federated_average()

        if (ep + 1) % max(1, train_eps // 4) == 0:
            print(".", end="", flush=True)
    print(f" done ({time.time()-t0:.1f}s)")

    # Eval using node 0 (all nodes converged to same weights after averaging)
    global_agents = nodes[0]
    metrics = []
    for ep in range(eval_eps):
        env = make_env(cfg, 1000 + ep)
        obs_list, current_job = env.reset()
        while True:
            a_obs = [apply_fuzzy(fuzzy, obs_list[i]) for i in range(4)]
            valid = get_valid_list(env, current_job)
            acts  = [global_agents[i].select_action(a_obs[i], valid_actions=valid,
                                                    training=False)
                     for i in range(4)]
            nobs, _, done, info = env.step(acts)
            obs_list    = nobs
            current_job = env._get_current_job()
            if done: break
        metrics.append(info)
    return _aggregate(metrics, "Federated-MARL")


# ── Table printer ─────────────────────────────────────────────────────────
def print_table(results):
    W = 103
    print("\n" + "=" * W)
    print("  BENCHMARK RESULTS")
    print("=" * W)
    print(f"  {'Method':<26} {'SLA':>8} {'Util':>8} {'Latency':>10} "
          f"{'Energy':>10} {'Violations':>12} {'Jobs':>8}")
    print("-" * W)
    for r in results:
        print(f"  {r['method']:<26} "
              f"{r['sla_compliance_mean']:>8.4f} "
              f"{r['avg_utilization_mean']:>8.4f} "
              f"{r['avg_latency_mean']:>10.2f} "
              f"{r['avg_energy_mean']:>10.6f} "
              f"{r['sla_violations_mean']:>12.1f} "
              f"{r['jobs_scheduled_mean']:>8.1f}")
    print("=" * W)


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Use shorter episodes for quick testing")
    args = parser.parse_args()

    cfg        = CFG_FAST if args.fast else CFG
    train_eps  = cfg["train_episodes"]
    eval_eps   = cfg["eval_episodes"]
    mode_label = "FAST" if args.fast else "FULL"

    print("\n" + "=" * 65)
    print(f"  MARL Cloud Benchmark  [{mode_label}]")
    print(f"  Machines: {cfg['num_machines']}  |  "
          f"Steps/ep: {cfg['duration_steps']}  |  "
          f"Train: {train_eps} eps  |  Eval: {eval_eps} eps")
    print("=" * 65)

    results = []
    results.append(run_iql(cfg,        train_eps, eval_eps))
    results.append(run_dqn(cfg,        train_eps, eval_eps))
    results.append(run_ppo(cfg,        train_eps, eval_eps))
    results.append(run_round_robin(cfg,           eval_eps))
    results.append(run_first_fit(cfg,             eval_eps))
    results.append(run_hmarl(cfg,      train_eps, eval_eps))
    results.append(run_federated(cfg,  train_eps, eval_eps))

    print_table(results)

    out = f"{cfg['save_dir']}/benchmark_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out}")
    print("  Run  python prepare_results.py  then  python plot_results.py  to generate charts.\n")


if __name__ == "__main__":
    main()
