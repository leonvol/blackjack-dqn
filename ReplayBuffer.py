import numpy as np
import random


class ReplayBuffer:
    def __init__(self, max_len):
        self.max_len = max_len
        self.buffer = []
        self.count = 0

    def push(self, data):
        if self.count == self.max_len:
            del self.buffer[np.random.randint(0, self.max_len)]
            self.count -= 1
        self.buffer.append(data)
        self.count += 1

    def push_all(self, data):
        diff = min(self.max_len - self.count - len(data), 0)
        for _ in range(diff):
            self.buffer.pop(random.randrange(self.count))
            self.count -= 1
        self.buffer.extend(data)
        self.count += len(data)

    def sample(self, batch):
        return random.sample(self.buffer, batch)
