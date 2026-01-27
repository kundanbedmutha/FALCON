"""
train.py — Train IQL MARL with Fuzzy + LLM Reward Shaping
Run:           python train.py
Run ablation:  python train.py --mode ablation
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import json
from env.cloud_env import CloudEnv
from agents.iql_agent import IQLAgent
from fuzzy.fuzzy_preprocessor import FuzzyPreprocessor
from llm.reward_shaper import LLMRewardShaper

CFG = {
    "num_machines":   20,
    "duration_steps": 2000,
    "episodes":       30,
    "seed":           42,
    "lr":             1e-3,
    "gamma":          0.99,
    "batch_size":     64,
    "hidden":         256,
    "use_fuzzy":      True,
    "use_llm_reward": True,
    "save_dir":       "results",
}

os.makedirs(CFG["save_dir"], exist_ok=True)


def make_env(seed_offset=0):
    return CloudEnv(num_machines=CFG["num_machines"],
                    seed=CFG["seed"] + seed_offset,
                    duration_steps=CFG["duration_steps"])


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


def shape_reward(raw_reward, info, llm, step):
    w       = llm.update(info, step)
    util    = float(info.get("avg_utilization", 0.5))
    sla     = float(info.get("sla_compliance",  1.0))
    energy  = float(min(info.get("avg_energy",  0.1), 1.0))
    latency = float(min(info.get("avg_latency", 1.0) / 200.0, 1.0))
    shaped  = (w["w_util"] * util + w["w_sla"] * sla
               - w["w_energy"] * energy - w["w_latency"] * latency)
    return float(0.6 * shaped + 0.4 * raw_reward)


def make_agents(obs_dim):
    return [IQLAgent(agent_id=i, obs_dim=obs_dim,
                     action_dim=CFG["num_machines"],
                     hidden=CFG["hidden"], lr=CFG["lr"],
                     gamma=CFG["gamma"], batch_size=CFG["batch_size"],
                     seed=CFG["seed"])
            for i in range(CloudEnv.N_AGENTS)]


def run_episode(env, agents, fuzzy, llm,
                use_fuzzy=True, use_llm=True, training=True):
    obs_list, current_job = env.reset()
    ep_rewards = [[] for _ in range(CloudEnv.N_AGENTS)]
    ep_losses  = [[] for _ in range(CloudEnv.N_AGENTS)]
    step = 0
    while True:
        a_obs = ([apply_fuzzy(fuzzy, obs_list[i]) for i in range(CloudEnv.N_AGENTS)]
                 if use_fuzzy
                 else [np.asarray(obs_list[i], dtype=np.float32) for i in range(CloudEnv.N_AGENTS)])

        valid   = get_valid_list(env, current_job)
        actions = [agents[i].select_action(a_obs[i], valid_actions=valid, training=training)
                   for i in range(CloudEnv.N_AGENTS)]

        nobs, rewards, done, info = env.step(actions)

        if use_llm:
            rewards = [shape_reward(rewards[i], info, llm, step)
                       for i in range(CloudEnv.N_AGENTS)]

        na_obs = ([apply_fuzzy(fuzzy, nobs[i]) for i in range(CloudEnv.N_AGENTS)]
                  if use_fuzzy
                  else [np.asarray(nobs[i], dtype=np.float32) for i in range(CloudEnv.N_AGENTS)])

        if training:
            for i in range(CloudEnv.N_AGENTS):
                agents[i].store(a_obs[i], actions[i], rewards[i], na_obs[i], float(done))
                loss = agents[i].learn()
                if loss is not None:
                    ep_losses[i].append(loss)

        for i in range(CloudEnv.N_AGENTS):
            ep_rewards[i].append(rewards[i])

        obs_list    = nobs
        current_job = env._get_current_job()
        step += 1
        if done:
            break

    return ([float(np.mean(r)) if r else 0.0 for r in ep_rewards],
            [float(np.mean(l)) if l else 0.0 for l in ep_losses],
            info)


def train_variant(tag, use_fuzzy, use_llm, episodes, save_history=True):
    env   = make_env()
    fuzzy = FuzzyPreprocessor()
    llm   = LLMRewardShaper()
    obs_dim = env.obs_size + (fuzzy.extra_dims if use_fuzzy else 0)
    agents  = make_agents(obs_dim)

    history = {k: [] for k in ["episode","avg_reward","sla_compliance",
                                "avg_utilization","avg_latency","avg_energy",
                                "sla_violations","jobs_scheduled","llm_profile"]}
    best_sla = 0.0

    for ep in range(episodes):
        _, _, info = run_episode(make_env(ep), agents, fuzzy, llm,
                                 use_fuzzy, use_llm, training=True)
        rewards_ep, _, info = run_episode(make_env(ep), agents, fuzzy, llm,
                                          use_fuzzy, use_llm, training=True)
        mean_r  = float(np.mean(rewards_ep))
        profile = llm.get_profile() if use_llm else "N/A"

        history["episode"].append(ep + 1)
        history["avg_reward"].append(round(mean_r, 4))
        history["sla_compliance"].append(round(info["sla_compliance"], 4))
        history["avg_utilization"].append(round(info["avg_utilization"], 4))
        history["avg_latency"].append(round(info["avg_latency"], 4))
        history["avg_energy"].append(round(info["avg_energy"], 6))
        history["sla_violations"].append(info["sla_violations"])
        history["jobs_scheduled"].append(info["jobs_scheduled"])
        history["llm_profile"].append(profile)

        if info["sla_compliance"] > best_sla:
            best_sla = info["sla_compliance"]

        print(f"  Ep {ep+1:3d}/{episodes} | Reward: {mean_r:+.4f} | "
              f"SLA: {info['sla_compliance']:.3f} | Util: {info['avg_utilization']:.3f} | "
              f"Lat: {info['avg_latency']:.1f} | Viol: {info['sla_violations']:4d} | "
              f"Jobs: {info['jobs_scheduled']:5d} | [{profile}]")

    print(f"\n  Training complete. Best SLA: {best_sla:.4f}")
    if save_history:
        with open(f"{CFG['save_dir']}/iql_training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        with open(f"{CFG['save_dir']}/iql_agent_stats.json", "w") as f:
            json.dump([agents[i].get_metrics() for i in range(CloudEnv.N_AGENTS)], f, indent=2)
        print(f"  Saved to {CFG['save_dir']}/")
    return agents, history


def run_ablation(episodes):
    print("\n" + "=" * 60)
    print("  ABLATION STUDY")
    print("=" * 60)
    abl_results = {}
    for tag, use_fuzzy, use_llm in [
        ("IQL_Base",      False, False),
        ("IQL+Fuzzy",     True,  False),
        ("IQL+Fuzzy+LLM", True,  True),
    ]:
        print(f"\n  ── {tag}  (fuzzy={use_fuzzy}, llm={use_llm})")
        agents, _ = train_variant(tag, use_fuzzy, use_llm, episodes,
                                  save_history=(tag == "IQL+Fuzzy+LLM"))
        fuzzy = FuzzyPreprocessor(); llm_e = LLMRewardShaper()
        metrics = [run_episode(make_env(1000+ep), agents, fuzzy, llm_e,
                               use_fuzzy, use_llm, False)[2] for ep in range(5)]
        abl_results[tag] = {
            k: round(float(np.mean([m[k] for m in metrics])), 4)
            for k in ["sla_compliance","avg_utilization","avg_latency",
                      "avg_energy","sla_violations","jobs_scheduled"]
        }
    with open(f"{CFG['save_dir']}/ablation_results.json", "w") as f:
        json.dump(abl_results, f, indent=2)
    print(f"\n  ABLATION SUMMARY")
    print(f"  {'Variant':<20} {'SLA':>8} {'Util':>8} {'Latency':>10} {'Energy':>10}")
    print("  " + "-" * 60)
    for tag, v in abl_results.items():
        print(f"  {tag:<20} {v['sla_compliance']:>8.4f} {v['avg_utilization']:>8.4f} "
              f"{v['avg_latency']:>10.2f} {v['avg_energy']:>10.6f}")


def train():
    print("=" * 60)
    print("  MARL Cloud Resource Management — IQL Training")
    print("=" * 60)
    print(f"  Machines: {CFG['num_machines']}  |  Episodes: {CFG['episodes']}")
    print(f"  Fuzzy: {CFG['use_fuzzy']}  |  LLM Reward: {CFG['use_llm_reward']}")
    print("=" * 60)
    env   = make_env()
    fuzzy = FuzzyPreprocessor()
    obs_dim = env.obs_size + fuzzy.extra_dims
    print(f"  Full obs: {env.obs_size}  +  Fuzzy dims: {fuzzy.extra_dims}  =  {obs_dim}\n")
    train_variant("IQL+Fuzzy+LLM", CFG["use_fuzzy"], CFG["use_llm_reward"],
                  CFG["episodes"], save_history=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "ablation"], default="train")
    args = parser.parse_args()
    if args.mode == "ablation":
        run_ablation(CFG["episodes"])
    else:
        train()
