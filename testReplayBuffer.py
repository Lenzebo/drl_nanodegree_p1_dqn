from ExperienceReplayBuffer import ExperienceReplayBuffer, PrioritizedReplayBuffer
import numpy as np

state_size = [10]

buffer = PrioritizedReplayBuffer(action_size=3, buffer_size=10, alpha=0.8, device="cuda:0", seed=2773)
# buffer = ExperienceReplayBuffer(action_size=3, buffer_size=10, device = "cuda:0", seed=2773)

# lets fill the experience buffer with at least batch_size
for i in range(10):
    buffer.add(i * np.ones(state_size), i % 3, 25 * i, (i + 1) * np.ones(state_size), False)

buffer.update([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
              [1.0, 0.0001, 1.0, 1.0, 1.0, 0.0001, 0.0001, 1.0, 0.0001, 0.0001])

print(buffer.sample(6))
print(buffer.sample(5))
print(buffer.sample(4))
