from DQNAgent import Agent, HyperParameter
from QNetwork import QNetwork
from ExperienceReplayBuffer import *

action_size = 3
state_size = 2
buffer = 100
batch = 2

params = HyperParameter(batch_size=batch)
memory = PrioritizedReplayBuffer(action_size=action_size, buffer_size=buffer, alpha=0, device="cuda:0", seed=10)
model = QNetwork(state_size, action_size, 123)
agent = Agent(10, 3, "cuda:0", memory, model, hyperparameter=params)

agent.step([0.0, 1.0], 1, 0, [0.5, 0.5], False)
agent.step([0.0, 1.0], 1, 0, [0.5, 0.5], False)
agent.step([0.0, 1.0], 1, 0, [0.5, 0.5], False)
agent.step([0.0, 1.0], 1, 0, [0.5, 0.5], False)
agent.step([0.0, 1.0], 1, 0, [0.5, 0.5], False)
agent.step([0.0, 1.0], 1, 0, [0.5, 0.5], False)
