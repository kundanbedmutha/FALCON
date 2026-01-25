"""
llm/reward_shaper.py
LLM-based dynamic reward shaping.

Simulates an LLM that reads current cloud telemetry and switches
reward weight profiles based on operational context:
  - BALANCED      : default equal weighting
  - SLA_PRIORITY  : heavy SLA penalty (peak hours / high violations)
  - ENERGY_SAVE   : reduce energy cost (off-peak / green mode)
  - THROUGHPUT    : maximise utilisation (burst workloads)
  - LATENCY_FOCUS : minimise latency (interactive workloads)

In a production system, the update() call would query a real LLM API.
Here we use deterministic rule-based switching that mirrors what an LLM
would reason about given the same telemetry — fully reproducible and
fast, while matching the paper's description exactly.
"""

import numpy as np


# ── Reward weight profiles ────────────────────────────────────────────────
PROFILES = {
    "BALANCED": {
        "w_util":    0.30,
        "w_sla":     0.30,
        "w_energy":  0.20,
        "w_latency": 0.20,
    },
    "SLA_PRIORITY": {
        "w_util":    0.15,
        "w_sla":     0.55,
        "w_energy":  0.15,
        "w_latency": 0.15,
    },
    "ENERGY_SAVE": {
        "w_util":    0.25,
        "w_sla":     0.25,
        "w_energy":  0.40,
        "w_latency": 0.10,
    },
    "THROUGHPUT": {
        "w_util":    0.50,
        "w_sla":     0.20,
        "w_energy":  0.15,
        "w_latency": 0.15,
    },
    "LATENCY_FOCUS": {
        "w_util":    0.20,
        "w_sla":     0.25,
        "w_energy":  0.10,
        "w_latency": 0.45,
    },
}

# How long to stay in a profile before re-evaluating (steps)
PROFILE_HOLD_STEPS = 200


class LLMRewardShaper:
    """
    Mimics an LLM that reads cloud telemetry every PROFILE_HOLD_STEPS
    steps and returns updated reward weights.

    Switching logic (what the LLM would reason):
      - SLA compliance < 0.45  → SLA_PRIORITY
      - avg_energy > 2.8       → ENERGY_SAVE
      - avg_utilization < 0.60 → THROUGHPUT
      - avg_latency > 100      → LATENCY_FOCUS
      - else                   → BALANCED
    """

    def __init__(self):
        self._profile      = "BALANCED"
        self._weights      = PROFILES["BALANCED"].copy()
        self._hold_counter = 0
        self._history      = []   # list of (step, profile)

    # ── Public API ────────────────────────────────────────────────────────
    def update(self, info: dict, step: int) -> dict:
        """
        Called every environment step with the latest info dict.
        Returns current reward weights.
        """
        self._hold_counter += 1
        if self._hold_counter >= PROFILE_HOLD_STEPS:
            self._hold_counter = 0
            new_profile = self._select_profile(info)
            if new_profile != self._profile:
                self._profile = new_profile
                self._weights = PROFILES[new_profile].copy()
                self._history.append((step, new_profile))
        return self._weights.copy()

    def get_profile(self) -> str:
        return self._profile

    def get_history(self) -> list:
        return list(self._history)

    def get_weights(self) -> dict:
        return self._weights.copy()

    # ── Internal switching logic ──────────────────────────────────────────
    def _select_profile(self, info: dict) -> str:
        sla     = float(info.get("sla_compliance",  0.5))
        energy  = float(info.get("avg_energy",      2.0))
        util    = float(info.get("avg_utilization", 0.7))
        latency = float(info.get("avg_latency",     80.0))

        # Priority order: SLA first → energy → throughput → latency → balanced
        if sla < 0.45:
            return "SLA_PRIORITY"
        if energy > 2.8:
            return "ENERGY_SAVE"
        if util < 0.60:
            return "THROUGHPUT"
        if latency > 100.0:
            return "LATENCY_FOCUS"
        return "BALANCED"
