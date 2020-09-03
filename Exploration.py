from abc import ABC, abstractmethod
import numpy as np
import random


def _create_valid_list(n_actions, mask):
    """Create a list of the indexes of True occurences in mask"""
    if mask is None:
        return [i for i, _ in enumerate(range(n_actions))]
    return [i for i, val in enumerate(mask) if val]


class Exploration(ABC):
    @abstractmethod
    def choose_action(self, model, x, step, n_actions, mask=None):
        pass

    @abstractmethod
    def should_predict(self, step):
        pass

    @abstractmethod
    def choose_random(self, n_actions, mask=None):
        pass


class EpsilonGreedy(Exploration):

    def __init__(self, epsilon=0.5, decay=0.009, epsilon_end=0, end_episode=500000):
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.decay = decay
        self.end_episode = end_episode

    def choose_action(self, model, x, step, n_actions, mask=None):
        """randomly chooses action based on epsilon greedy algorithm"""
        if self.should_predict(step):
            return self.choose_random(n_actions, mask)

        return np.argmax(model(np.array([x]) * mask))

    def should_predict(self, step):
        if self.end_episode > step:
            return random.uniform(0, 1) > max(self.epsilon ** (self.decay * step), self.epsilon_end)
        return True

    def choose_random(self, n_actions, mask=None):
        return random.choice(_create_valid_list(n_actions, mask))


# Purely random policy
class RandomPolicy(Exploration):

    def choose_action(self, model, x, step, n_actions, mask=None):
        return self.choose_random(n_actions, mask)

    def should_predict(self, step):
        return True

    def choose_random(self, n_actions, mask=None):
        return random.choice(_create_valid_list(n_actions, mask))
