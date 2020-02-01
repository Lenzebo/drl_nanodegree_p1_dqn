import random
import torch
from torch import optim
import torch.nn.functional as F

from collections import namedtuple
import numpy as np

HyperParameter = namedtuple("HyperParameter",
                            field_names=["learning_rate", "update_every", "batch_size", "gamma", "tau"])
HyperParameter.__new__.__defaults__ = (5e-4, 4, 64, 0.99, 1e-3)


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, device, memory, network,
                 hyperparameter: HyperParameter = HyperParameter()):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            device (str): device identifier used for the network
            memory: replay memory implementation
            network: QNetwork implementation
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.hyperparameter = hyperparameter

        # Q-Network
        self.qnetwork_local = network
        self.qnetwork_target = network.clone()

        self.qnetwork_local = self.qnetwork_local.to(device)
        self.qnetwork_target = self.qnetwork_target.to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.hyperparameter.learning_rate)

        # Replay memory
        self.memory = memory
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.hyperparameter.update_every

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.hyperparameter.batch_size:
                experiences = self.memory.sample(self.hyperparameter.batch_size)
                self.learn(experiences, self.hyperparameter.gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, idxs, importance = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Update the memory and tell it how good or bad one memory was
        self.memory.update(idxs, Q_expected - Q_targets)

        # Compute loss
        loss = torch.sum((importance * (Q_expected - Q_targets)) ** 2)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.hyperparameter.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
