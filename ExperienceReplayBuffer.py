import numpy as np
import random
import torch
from collections import namedtuple, deque
from operator import itemgetter
from recordtype import recordtype


class ExperienceReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, device, seed):
        """Initialize a ExperienceReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            seed (int): random seed
        """
        self.device = device
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def update(self, index, error):
        pass

    def sample_indices(self, batch_size):
        idxs = np.random.choice(len(self.memory), batch_size, replace=False)
        importance = np.ones_like(idxs, dtype=np.float)
        return idxs, importance

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        index, importance = self.sample_indices(batch_size)
        experiences = [self.memory[idx] for idx in index]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        importance = torch.from_numpy(importance).float().to(self.device)

        return (states, actions, rewards, next_states, dones, index, importance)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PrioritizedReplayBuffer(ExperienceReplayBuffer):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, alpha, device, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            seed (int): random seed
        """
        super(PrioritizedReplayBuffer, self).__init__(action_size, buffer_size, device, seed)
        self.experience = recordtype("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done", "priority"])
        self.alpha = alpha
        self.beta = 1.0
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, self.get_max_priority())
        self.memory.append(e)

    def get_max_priority(self):
        return self.max_priority

    def update(self, index, error):

        # if we got a torch tensor, we copy it to cpu and create a numpy array from it
        if type(error) is torch.Tensor:
            error = error.cpu().detach().numpy().reshape([-1])

        try:
            for i in range(len(index)):
                prio = np.power(np.abs(error[i]), self.alpha)
                self.memory[index[i]].priority = prio
                self.max_priority = max(self.max_priority, prio)
        except TypeError:  # i index is not iterable
            prio = np.power(np.abs(error), self.alpha)
            self.memory[index].priority = prio
            self.max_priority = max(self.max_priority, prio)

    def sample_indices(self, batch_size):
        probabilities = np.asarray([e.priority for e in self.memory])
        probabilities /= probabilities.sum()
        index = np.random.choice(len(self.memory), batch_size, replace=False, p=probabilities)
        probabilities = probabilities[index]

        importance = np.power(batch_size * probabilities, -self.beta)
        importance /= importance.max()
        return index, importance
