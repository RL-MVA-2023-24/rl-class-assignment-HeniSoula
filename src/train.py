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
        device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

        # observation_tensor = torch.from_numpy(observation).view(1, -1).float().to(device)
        # print(observation_tensor.shape)
        # action_values = self.model(observation_tensor)
        # print("owo")
        # a = torch.argmax(action_values.view(-1)).item()
        # print(type(a))
        # return a

    def save(self, path):
        pass

    def load(self):
        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n
        nb_neurons = 24
        self.model = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons), nn.ReLU(), nn.Linear(nb_neurons, n_action))
        self.model = torch.load("/home/onyxia/work/rl-class-assignment-HeniSoula/src/models/DQN1.pt").to(device)
        self.model.eval()

        # payload = {
        #     'n_states':self.state_size,
        #     'n_hidden':self.hidden_size,
        #     'n_actions': self.action_size,

        # }
        # with open(self.payload_path, 'wb') as file:
        #     pkl.dump(payload, file)
        # torch.save(self.network.state_dict(), self.model_path)
