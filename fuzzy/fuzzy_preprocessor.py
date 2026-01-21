"""
fuzzy/fuzzy_preprocessor.py
Fuzzy logic preprocessing for cloud workload observations.

Takes first 8 dims of the obs vector as linguistic inputs:
  [cpu_util, mem_util, net_util, n_jobs_norm, load_norm,
   job_cpu_req, job_mem_req, job_net_req]

Outputs the original 8 dims + 14 fuzzy membership features = 22 total.
train.py slices [8:] to get the 14 new dims, then concatenates:
  final_obs = [raw_105_dims | fuzzy_14_dims] = 119 dims
"""

import numpy as np

try:
    import skfuzzy as fuzz
    SKFUZZY_OK = True
except ImportError:
    SKFUZZY_OK = False


class FuzzyPreprocessor:
    """
    Produces 14 fuzzy membership values appended after the raw obs.
    extra_dims = 14
    """

    extra_dims = 14

    def __init__(self):
        # Universe of discourse [0, 1] for all variables
        self._u = np.linspace(0, 1, 100)

        # Define membership functions for cpu, mem, load (low/med/high)
        self._mf = {
            "low":  fuzz.trimf(self._u, [0.0, 0.0, 0.5]) if SKFUZZY_OK else None,
            "med":  fuzz.trimf(self._u, [0.2, 0.5, 0.8]) if SKFUZZY_OK else None,
            "high": fuzz.trimf(self._u, [0.5, 1.0, 1.0]) if SKFUZZY_OK else None,
        }

        # Job size: small/medium/large
        self._mf_job = {
            "small":  fuzz.trimf(self._u, [0.0, 0.0, 0.3]) if SKFUZZY_OK else None,
            "medium": fuzz.trimf(self._u, [0.1, 0.3, 0.6]) if SKFUZZY_OK else None,
            "large":  fuzz.trimf(self._u, [0.4, 1.0, 1.0]) if SKFUZZY_OK else None,
        }

    def _membership(self, value, mf_dict):
        """Compute membership values for a crisp input."""
        value = float(np.clip(value, 0.0, 1.0))
        if not SKFUZZY_OK:
            # Fallback: simple piecewise linear without skfuzzy
            low  = max(0.0, 1.0 - value / 0.5) if value <= 0.5 else 0.0
            med  = max(0.0, 1.0 - abs(value - 0.5) / 0.3)
            high = max(0.0, (value - 0.5) / 0.5) if value >= 0.5 else 0.0
            return [low, med, high]

        idx = int(np.clip(value * 99, 0, 99))
        return [
            float(mf_dict["low"][idx]),
            float(mf_dict["med"][idx]),
            float(mf_dict["high"][idx]),
        ]

    def _job_membership(self, value):
        value = float(np.clip(value, 0.0, 1.0))
        if not SKFUZZY_OK:
            small  = max(0.0, 1.0 - value / 0.3)
            medium = max(0.0, 1.0 - abs(value - 0.3) / 0.3)
            large  = max(0.0, (value - 0.4) / 0.6)
            return [small, medium, large]
        idx = int(np.clip(value * 99, 0, 99))
        return [
            float(self._mf_job["small"][idx]),
            float(self._mf_job["medium"][idx]),
            float(self._mf_job["large"][idx]),
        ]

    def process(self, obs_chunk):
        """
        obs_chunk: array of 8 floats
          [cpu_util, mem_util, net_util, n_jobs_norm, load_norm,
           job_cpu_req, job_mem_req, job_net_req]

        Returns: array of 22 floats (original 8 + 14 fuzzy)
        """
        obs_chunk = np.asarray(obs_chunk, dtype=np.float32)
        cpu_util  = float(obs_chunk[0]) if len(obs_chunk) > 0 else 0.5
        mem_util  = float(obs_chunk[1]) if len(obs_chunk) > 1 else 0.5
        load_norm = float(obs_chunk[4]) if len(obs_chunk) > 4 else 0.5
        job_cpu   = float(obs_chunk[5]) if len(obs_chunk) > 5 else 0.2
        job_mem   = float(obs_chunk[6]) if len(obs_chunk) > 6 else 0.2

        # 3 + 3 + 3 + 3 + 2 = 14 fuzzy features
        f_cpu  = self._membership(cpu_util,  self._mf)      # 3
        f_mem  = self._membership(mem_util,  self._mf)      # 3
        f_load = self._membership(load_norm, self._mf)      # 3
        f_jcpu = self._job_membership(job_cpu)              # 3
        # Composite: overload risk (2 values: safe / at_risk)
        overload = min(1.0, cpu_util + mem_util)
        f_risk   = [max(0.0, 1.0 - overload / 0.7),        # safe
                    max(0.0, (overload - 0.5) / 0.5)]       # at_risk

        fuzzy_feats = np.array(
            f_cpu + f_mem + f_load + f_jcpu + f_risk,
            dtype=np.float32
        )  # 14 dims

        return np.concatenate([obs_chunk, fuzzy_feats]).astype(np.float32)  # 8+14=22
