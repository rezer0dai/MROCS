import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np

def rl_ibounds(layer):
    b = 1. / np.sqrt(layer.weight.data.size(0))
    return (-b, +b)

def initialize_weights(layer):
    if type(layer) not in [nn.Linear, ]:
        return
#    nn.init.kaiming_uniform_(layer.weight) # seems not performing well
    nn.init.uniform_(layer.weight.data, *rl_ibounds(layer))

class Critic(nn.Module):
    def __init__(self, n_rewards, state_size, action_size, fc1_units=400, fc2_units=300):
        super().__init__()

        self.fca = nn.Linear(action_size * n_rewards, state_size)
        self.fcs = nn.Linear(state_size * n_rewards, state_size * n_rewards)

        self.fc1 = nn.Linear(state_size * (1 + n_rewards), fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)#, bias=False)

        self.apply(initialize_weights)

        self.fc3 = nn.Linear(fc2_units, n_rewards)#, bias=False)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3) # seems this works better ? TODO : proper tests!!

    def forward(self, states, actions):
        assert 0 == actions.size(0) % states.size(0), "opla size mismatch!! {}::{}".format(actions.size(), states.size())
        delta = actions.size(0) // states.size(0)
        actions = torch.cat([
            actions[i*states.size(0):(i+1)*states.size(0), :] for i in range(delta)
            ], 1)

        # process so actions can contribute to Q-function more effectively ( theory .. )
        actions = self.fca(actions)
        states = F.relu(self.fcs(states))

        xs = torch.cat((states, actions), dim=1)
        x = F.relu(self.fc1(xs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x