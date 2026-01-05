"""
env/cloud_env.py
Cloud resource scheduling environment — pure NumPy, no torch required.

State per agent (105 dims total):
  - 4 agents × (machine_features + job_queue_features + global_stats)
  - Each agent sees: own-machine slice (3) + all-machine summary (20×5=100) + job_queue (5) = 105? 
    We match the train.py output: "Full obs: 105"

Observation layout (105 dims):
  machines:   20 × 5 = 100   (cpu_util, mem_util, net_util, running_jobs, load_norm)
  current_job: 5              (cpu_req, mem_req, net_req, priority, deadline_norm)
  Total: 105

Actions: choose one of 20 machines to assign current job.
"""

import numpy as np
from collections import deque


class CloudEnv:
    N_AGENTS  = 4
    N_MACHINE_FEATURES = 5
    N_JOB_FEATURES     = 5

    def __init__(self, num_machines=20, seed=42, duration_steps=2000):
        self.num_machines   = num_machines
        self.seed           = seed
        self.duration_steps = duration_steps
        self.rng            = np.random.RandomState(seed)

        # obs_size = machines × features + job_features
        self.obs_size = num_machines * self.N_MACHINE_FEATURES + self.N_JOB_FEATURES  # 105

        self.reset()

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────
    def _sample_job(self):
        cpu  = float(self.rng.uniform(0.02, 0.35))
        mem  = float(self.rng.uniform(0.02, 0.30))
        net  = float(self.rng.uniform(0.01, 0.20))
        pri  = float(self.rng.choice([1, 2, 3]))          # 1=low 3=high
        dur  = int(self.rng.randint(20, 300))              # steps
        ddl  = float(self.rng.uniform(0.3, 1.0))          # deadline fraction
        return {"cpu_req": cpu, "mem_req": mem, "net_req": net,
                "priority": pri / 3.0, "deadline_norm": ddl,
                "duration": dur, "age": 0, "sla_deadline": ddl * dur}

    def _get_obs(self):
        """Returns list of 4 identical obs vectors (each 105-dim)."""
        # Machine state (20×5)
        machine_flat = self.machines.flatten()  # 100 dims
        # Current job features (5)
        if self._current_job is not None:
            j = self._current_job
            job_feat = np.array([j["cpu_req"], j["mem_req"], j["net_req"],
                                  j["priority"], j["deadline_norm"]], dtype=np.float32)
        else:
            job_feat = np.zeros(self.N_JOB_FEATURES, dtype=np.float32)
        obs = np.concatenate([machine_flat, job_feat]).astype(np.float32)
        return [obs.copy() for _ in range(self.N_AGENTS)]

    def _get_current_job(self):
        return self._current_job

    # ──────────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────────
    def reset(self):
        # machines[:, 0]=cpu_used, 1=mem_used, 2=net_used, 3=n_jobs, 4=load_norm
        self.machines    = np.zeros((self.num_machines, self.N_MACHINE_FEATURES), dtype=np.float32)
        self._running    = {}   # machine_id → list of (remaining_steps, job)
        self._step_count = 0
        self._scheduled  = 0
        self._violations = 0
        self._total_latency = 0.0
        self._total_energy  = 0.0
        self._n_energy_samples = 0
        self._current_job = self._sample_job()
        return self._get_obs(), self._current_job

    # ──────────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────────
    def step(self, actions):
        """
        actions: list of 4 ints (machine index chosen by each agent).
        We use the majority-vote / first-agent decision for simplicity.
        """
        action = int(actions[0])          # primary decision from agent 0
        action = max(0, min(action, self.num_machines - 1))

        job = self._current_job
        reward_components = {}

        # ── Schedule job ──────────────────────────────────────────
        scheduled = False
        if job is not None:
            m = action
            if (self.machines[m, 0] + job["cpu_req"] <= 1.0 and
                    self.machines[m, 1] + job["mem_req"] <= 1.0 and
                    self.machines[m, 2] + job["net_req"] <= 1.0):
                self.machines[m, 0] += job["cpu_req"]
                self.machines[m, 1] += job["mem_req"]
                self.machines[m, 2] += job["net_req"]
                self.machines[m, 3] += 1
                if m not in self._running:
                    self._running[m] = []
                self._running[m].append([job["duration"], job])
                self._scheduled += 1
                scheduled = True
            else:
                # Forced: pick least loaded machine
                loads  = self.machines[:, 0] + self.machines[:, 1]
                fallback = int(np.argmin(loads))
                m = fallback
                self.machines[m, 0] = min(1.0, self.machines[m, 0] + job["cpu_req"])
                self.machines[m, 1] = min(1.0, self.machines[m, 1] + job["mem_req"])
                self.machines[m, 2] = min(1.0, self.machines[m, 2] + job["net_req"])
                self.machines[m, 3] += 1
                if m not in self._running:
                    self._running[m] = []
                self._running[m].append([job["duration"], job])
                self._scheduled += 1

            # Latency: proportional to load at time of scheduling
            latency = 50.0 + self.machines[m, 0] * 100.0 + self.rng.uniform(0, 20)
            self._total_latency += latency

            # SLA: check deadline
            if latency > job["sla_deadline"] * 150:
                self._violations += 1

        # ── Progress running jobs ─────────────────────────────────
        for mid in list(self._running.keys()):
            remaining_jobs = []
            for entry in self._running[mid]:
                entry[0] -= 1
                if entry[0] <= 0:
                    j = entry[1]
                    self.machines[mid, 0] = max(0.0, self.machines[mid, 0] - j["cpu_req"])
                    self.machines[mid, 1] = max(0.0, self.machines[mid, 1] - j["mem_req"])
                    self.machines[mid, 2] = max(0.0, self.machines[mid, 2] - j["net_req"])
                    self.machines[mid, 3] = max(0, int(self.machines[mid, 3]) - 1)
                else:
                    remaining_jobs.append(entry)
            if remaining_jobs:
                self._running[mid] = remaining_jobs
            else:
                del self._running[mid]

        # ── Update load_norm column ───────────────────────────────
        self.machines[:, 4] = (self.machines[:, 0] + self.machines[:, 1]) / 2.0

        # ── Energy: PUE-inspired model ────────────────────────────
        avg_cpu = float(np.mean(self.machines[:, 0]))
        energy  = 0.5 + avg_cpu * 2.0 + self.rng.uniform(0, 0.1)   # raw units
        self._total_energy  += energy
        self._n_energy_samples += 1

        # ── Reward ────────────────────────────────────────────────
        avg_util = float(np.mean(self.machines[:, 0] * 0.6 + self.machines[:, 1] * 0.4))
        sla_comp = 1.0 - self._violations / max(1, self._scheduled)
        norm_lat = min(latency if job else 80.0, 200.0) / 200.0
        norm_nrg = min(energy, 3.0) / 3.0

        reward = (0.4 * avg_util + 0.4 * sla_comp
                  - 0.1 * norm_lat - 0.1 * norm_nrg - 0.5)

        # ── Advance step ──────────────────────────────────────────
        self._step_count += 1
        self._current_job = self._sample_job() if self._step_count < self.duration_steps else None
        done = self._step_count >= self.duration_steps

        # ── Info dict ─────────────────────────────────────────────
        total_jobs = self._scheduled + self._violations
        info = {
            "sla_compliance":    float(sla_comp),
            "avg_utilization":   float(np.mean(self.machines[:, 0] * 0.6 + self.machines[:, 1] * 0.4)),
            "avg_latency":       float(self._total_latency / max(1, self._scheduled)),
            "avg_energy":        float(self._total_energy  / max(1, self._n_energy_samples)),
            "sla_violations":    int(self._violations),
            "jobs_scheduled":    int(self._scheduled),
        }

        rewards = [float(reward)] * self.N_AGENTS
        return self._get_obs(), rewards, done, info
