import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module): # share common preprocessing layer!
    # encoder could be : RNN, CNN, RBF, BatchNorm / GlobalNorm, others.. and combination of those
    def __init__(self, encoder, n_history, state_size, actor, critic):
        super().__init__()
        self.encoder = encoder
        self.actor = actor
        self.critic = critic

        self.n_history = n_history
        self.state_size = state_size

        for i, a in enumerate(self.actor):
            self.add_module("actor_%i"%i, a)
        for i, c in enumerate(self.critic):
            self.add_module("critic_%i"%i, c)

    def parameters(self):
        assert False, "should not be accessed!"

# TODO : where to make sense to train encoder -> at Actor, Critic, or both ??
    def actor_parameters(self):
        return np.concatenate([
#            list(self.encoder.parameters()),
            np.concatenate([list(actor.parameters()) for actor in self.actor], 0)])

    def critic_parameters(self, ind):
        c_i = ind if ind < len(self.critic) else 0
        return np.concatenate([
            list(self.encoder.parameters()),
            list(self.critic[c_i].parameters())])

    def forward(self, states, ind = 0):
        assert ind != -1 or len(self.critic) == 1, "you forgot to specify which critic should be used"

        states = torch.from_numpy(states)
        states = self.encoder(states.view(-1, self.state_size)).view(
                states.size(0), -1, self.n_history*self.state_size)

        actions = []
        for i in range(states.size(1)):
            a_i = i % len(self.actor)
            pi = self.actor[a_i](states[:, i, :])
            pi = torch.tanh(pi)
            actions.append(pi)
        actions = torch.cat(actions, 0)

        states = states.view(states.size(0), -1)
        return self.critic[ind](states, actions)

    def value(self, states, actions, ind):
        states = torch.from_numpy(states)
        states = self.encoder(states.view(-1, self.state_size)).view(states.size())
        return self.critic[ind](states, actions)

    def act(self, ind, states):
        ind = ind % len(self.actor)

        states = torch.from_numpy(states)
        states = self.encoder(states.view(-1, self.state_size)).view(states.size())

        pi = self.actor[ind](states)
        pi = torch.tanh(pi)
        return pi
