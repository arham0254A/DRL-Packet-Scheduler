from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

DEFAULT_ARRIVAL_RATES = (2.5, 6.0, 8.0)


class DynamicHetNetEnv(gym.Env):
    """
    Gymnasium Environment for QoS-Aware Packet Scheduling in Dynamic HetNets.

    Traffic Classes:
      - Class 0: URLLC / Voice (SLA Target: 2 time steps, Weight: 12.0)
      - Class 1: eMBB / Video (SLA Target: 6 time steps, Weight: 5.0)
      - Class 2: Best Effort / Data (SLA Target: 20 time steps, Weight: 1.0)

    Heterogeneous Links:
      - Link 0: 5G Sub-6GHz (Reliable macro link, moderate rate)
      - Link 1: 5G mmWave (High capacity, prone to dynamic line-of-sight blockage)
      - Link 2: Wi-Fi 6E (Unlicensed spectrum, dynamic contention noise)

    All randomness goes through self.np_random, so reset(seed=s) makes an
    episode's traffic and channel conditions fully reproducible.
    """

    def __init__(self, queue_capacity=40, episode_length=200):
        super().__init__()

        self.num_classes = 3
        self.num_links = 3
        self.queue_capacity = queue_capacity
        self.episode_length = episode_length

        # Action space: Discrete(9) -> Action ID = queue_idx * num_links + link_idx
        self.action_space = spaces.Discrete(self.num_classes * self.num_links)

        # State space: [q0..q2, hol0..hol2, arr0..arr2, c0..c2] (12 dimensions)
        # HOL delay is unbounded if a queue is starved, so there is no hard upper limit.
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(12,), dtype=np.float32
        )

        # SLA Latency Targets (in slots) and penalty weights
        self.sla_targets = [2.0, 6.0, 20.0]
        self.penalties = [12.0, 5.0, 1.0]

        # Blockage is an external condition set by the user, so it survives
        # episode resets. Clear it explicitly with set_mmwave_blockage(False).
        self.mmwave_blocked = False
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.queues = [deque(maxlen=self.queue_capacity) for _ in range(self.num_classes)]
        self.current_step = 0
        self.last_arrivals = [0, 0, 0]
        self.link_capacities = [10.0, 25.0, 15.0]

        return self._get_obs(), {}

    def set_mmwave_blockage(self, blocked: bool):
        self.mmwave_blocked = bool(blocked)

    def _update_link_capacities(self):
        """Simulates dynamic physical channel fading and blockages."""
        rng = self.np_random

        # Link 0 (5G Sub-6GHz): Rayleigh fading around mean 10.0
        c0 = max(2.0, 10.0 + rng.normal(0, 1.5))

        # Link 1 (5G mmWave): High rate (25.0), but blockage drastically drops capacity.
        # The random blockage roll is always drawn so the RNG stream stays aligned
        # whether or not the user has forced a blockage.
        random_blockage = rng.random() < 0.12
        if self.mmwave_blocked or random_blockage:
            c1 = max(0.0, rng.normal(1.5, 0.5))
        else:
            c1 = max(5.0, 25.0 + rng.normal(0, 4.0))

        # Link 2 (Wi-Fi 6E): Medium capacity (15.0) with dynamic contention noise
        c2 = max(1.0, 15.0 + rng.normal(0, 3.5))

        self.link_capacities = [float(c0), float(c1), float(c2)]

    def _get_obs(self):
        q_lens = [float(len(q)) for q in self.queues]
        hol_delays = [float(self.current_step - q[0]) if q else 0.0 for q in self.queues]
        obs = q_lens + hol_delays + [float(a) for a in self.last_arrivals] + self.link_capacities
        return np.array(obs, dtype=np.float32)

    def step(self, action, arrival_rates=None):
        self.current_step += 1

        # Decode action index into queue selection and link assignment
        queue_idx, link_idx = divmod(int(action), self.num_links)

        # Update dynamic channel conditions
        self._update_link_capacities()
        available_capacity = int(self.link_capacities[link_idx])

        if arrival_rates is None:
            arrival_rates = DEFAULT_ARRIVAL_RATES

        # Generate Poisson packet arrivals
        drops = [0, 0, 0]
        for k in range(self.num_classes):
            arr = int(self.np_random.poisson(arrival_rates[k]))
            self.last_arrivals[k] = arr
            accepted = min(arr, self.queue_capacity - len(self.queues[k]))
            self.queues[k].extend([self.current_step] * accepted)
            drops[k] = arr - accepted  # Queue Overflow

        # Serve selected queue over selected link
        served_delays = []
        target_q = self.queues[queue_idx]
        while target_q and len(served_delays) < available_capacity:
            served_delays.append(self.current_step - target_q.popleft())
        served_count = len(served_delays)

        # Compute QoS Multi-Objective Reward
        reward = float(served_count * 2.0)

        # SLA Latency Violation Penalty
        sla_violations = [0, 0, 0]
        for k in range(self.num_classes):
            if self.queues[k]:
                hol_delay = self.current_step - self.queues[k][0]
                reward -= self.penalties[k] * max(0.0, hol_delay - self.sla_targets[k])
                if hol_delay > self.sla_targets[k]:
                    sla_violations[k] = 1

        # Buffer Overflow Drop Penalty
        for k in range(self.num_classes):
            reward -= self.penalties[k] * 5.0 * drops[k]

        # The episode ends on a time limit, not because the system reached a
        # terminal state, so it is reported as truncated (Gymnasium convention).
        truncated = self.current_step >= self.episode_length

        info = {
            "queue_idx": queue_idx,
            "link_idx": link_idx,
            "served_count": served_count,
            "drops": drops,
            "avg_served_delay": float(np.mean(served_delays)) if served_delays else 0.0,
            "sla_violations": sla_violations,
            "link_capacities": list(self.link_capacities),
        }

        return self._get_obs(), reward, False, truncated, info
