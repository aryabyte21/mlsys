import random

class AdaptiveSpeculativeDecoder:
    def __init__(self, k_min=3, k_max=8, k_init=4, window=10):
        self.k = k_init
        self.k_min = k_min
        self.k_max = k_max
        self.window = window
        # rolling history: 1 = accepted, 0 = rejected
        self.history = []

    def update(self, accepted: int, proposed: int):
        # record per-token outcomes
        self.history.extend([1] * accepted + [0] * (proposed - accepted))
        # keep only last N
        self.history = self.history[-self.window:]

        alpha = sum(self.history) / len(self.history)

        if alpha > 0.8 and self.k < self.k_max:
            self.k += 1
        elif alpha < 0.5 and self.k > self.k_min:
            self.k -= 1

        return self.k, alpha

    def get_k(self):
        return self.k


# You can import this class in chat_engine.py to track acceptance rates over time.
# decoder = AdaptiveSpeculativeDecoder()

class DraftTemperatureOptimizer:
    def __init__(self):
        # track acceptance rate per temperature bucket
        self.temp_stats = {
            0.0: {"accepted": 0, "proposed": 0},
            0.3: {"accepted": 0, "proposed": 0},
            0.5: {"accepted": 0, "proposed": 0},
            0.7: {"accepted": 0, "proposed": 0},
        }
        self.current_temp = 0.3
        self.explore_every = 20  # explore a random temp every 20 requests
        self.request_count = 0

    def get_temperature(self):
        self.request_count += 1
        # epsilon-greedy exploration
        if self.request_count % self.explore_every == 0:
            self.current_temp = float(
                random.choice(list(self.temp_stats.keys()))
            )
        return self.current_temp

    def update(self, temp, accepted, proposed):
        if temp in self.temp_stats:
            self.temp_stats[temp]["accepted"] += accepted
            self.temp_stats[temp]["proposed"] += proposed

        # pick best temp seen so far
        best_temp = max(
            self.temp_stats,
            key=lambda t: (
                self.temp_stats[t]["accepted"] /
                max(self.temp_stats[t]["proposed"], 1)
            )
        )
        self.current_temp = best_temp
        return best_temp

