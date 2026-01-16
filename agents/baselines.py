"""
agents/baselines.py
Rule-based baseline agents.
"""

import numpy as np


class RoundRobinAgent:
    def __init__(self, num_machines):
        self.num_machines = num_machines
        self._counter     = 0

    def select_action(self):
        action = self._counter % self.num_machines
        self._counter += 1
        return int(action)

    def store(self, *args, **kwargs):
        pass  # stateless


class FirstFitAgent:
    """Assigns job to the first machine with enough capacity."""

    def __init__(self, num_machines):
        self.num_machines = num_machines

    def select_action(self, obs, machines=None, job=None):
        if machines is None or job is None:
            return 0

        for m in range(self.num_machines):
            if (machines[m, 0] + job["cpu_req"] <= 1.0 and
                    machines[m, 1] + job["mem_req"] <= 1.0 and
                    machines[m, 2] + job["net_req"] <= 1.0):
                return int(m)
        # Fallback: least loaded
        loads = machines[:, 0] + machines[:, 1]
        return int(np.argmin(loads))

    def store(self, *args, **kwargs):
        pass
