import random
from collections import deque

import numpy as np


class ReplayMemory(deque):
    def __init__(self, capacity, seed=None):
        super().__init__(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, state, action, reward, next_state, done):
        self.append((state, action, reward, next_state, done))

    def bulk_push(self, samples):
        for sample in samples:
            assert len(sample) == 5
        self.extend(samples)

    def sample(self, batch_size):
        batch = self.rng.sample(self, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
