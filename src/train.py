from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch.nn as nn
import torch

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

device = "cuda"


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!


class ProjectAgent:
    def act(self, observation, use_random=False):
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0))
            return torch.argmax(Q).item()

    def save(self, path):
        pass

    def load(self):
        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n
        nb_neurons=128
        self.model = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
                                nn.ReLU(),
                                nn.Linear(nb_neurons, nb_neurons),
                                nn.ReLU(), 
                                nn.Linear(nb_neurons, n_action))
        self.model = torch.load("src/models/DQN7.pt", map_location=torch.device('cpu'))
        self.model.eval()
