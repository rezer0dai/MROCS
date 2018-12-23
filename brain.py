import numpy as np
import random, copy, sys

import torch
import torch.nn.functional as F
import torch.optim as optim

torch.set_default_tensor_type('torch.DoubleTensor')

from critic import Critic

from nes import *

from memory import Memory as ReplayBuffer
#sys.path.append("PrioritizedExperienceReplay")
#from PrioritizedExperienceReplay.proportional import Experience as ReplayBuffer

PRIO_ALPHA = .8
PRIO_BETA = .9
PRIO_MIN = 1e-10
PRIO_MAX = 1e2

device = torch.device("cpu")#cuda" if torch.cuda.is_available() else "cpu")

from ac import *

from critic import *

class Brain():
    def __init__(self,
            Encoder,
            n_rewards, detach, n_critics,
            n_history, state_size, action_size,
            learning_delay, learning_repeat, resample_delay,
            lr_actor, lr_critic, clip_norm,
            n_step, gamma, tau, soft_sync,
            buffer_size, batch_size):

        self.n_rewards = n_rewards
        self.clip_norm = clip_norm
        self.learning_delay, self.learning_repeat, self.resample_delay = learning_delay, learning_repeat, resample_delay
        self.n_step, self.gamma, self.tau, self.soft_sync = n_step, gamma, tau, soft_sync
        self.batch_size = batch_size

        self.n_discount = self.gamma ** self.n_step

        self.count = 0

        nes_layers = [n_history*state_size, 400, 300, action_size]
        nes = NoisyNetFactory(nes_layers)
                                     
        encoder = Encoder(state_size)
        encoder = encoder.share_memory()
        self.ac_explorer = ActorCritic(encoder, n_history, state_size,
                    [ nes.head() for _ in range(1 if not detach else n_rewards) ],
                    [ Critic(n_rewards, n_history*state_size, action_size) for _ in range(n_critics) ])

        self.ac_target = ActorCritic(encoder, n_history, state_size,
                    [ NoisyNetFactory(nes_layers).head() ],
                    [ Critic(n_rewards, n_history*state_size, action_size) for _ in range(n_critics) ])

        # sync
        for explorer in self.ac_explorer.actor:
            self._soft_update(self.ac_target.actor[0].parameters(), explorer, 1.)
        for i in range(n_critics):
            self._soft_update(self.ac_target.critic[i].parameters(), self.ac_explorer.critic[i], 1.)

        # set optimizers, RMSprop is also good choice !
        self.actor_optimizer = optim.Adam(self.ac_explorer.actor_parameters(), lr=lr_actor)
        self.critic_optimizer = [ optim.Adam(
            self.ac_explorer.critic_parameters(i), lr=lr_critic) for i in range(n_critics) ]

        # Replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size, PRIO_ALPHA)

    def resample(self, t):
        if 0 != t % self.resample_delay:# * self.learning_delay:
            return
        for actor in self.ac_explorer.actor:
            actor.sample_noise(random.randint(0, len(actor.layers)))

    def step(self, state, action, reward, next_state, t):
        self.resample(t)
        # batch experiences based on number of agents
        n_samples = len(state) // self.n_rewards
        state = state.reshape(n_samples, -1)
        action = action.reshape(n_samples, -1)
        reward = reward.reshape(n_samples, -1)
        next_state = next_state.reshape(n_samples, -1)
        # append to prio buffer with maximum priority
        for s, a, r, n in zip(state, action, reward, next_state):
            self.memory.add([ s, a, r, n ], 1.)

        if len(self.memory) < self.batch_size:
            return

        # skip x steps ~ give enough time to perform to get some feadback
        if 0 != t % self.learning_delay:
            return

        # postpone target + explorer sync ~ make sense with small steps only ?
        self.count += 1

        # learn
        losses = []
        for i in range(self.learning_repeat):
            tau = 0 if 0 != (i+self.count) % self.soft_sync else self.tau
            batch, _, inds = self.memory.select(PRIO_BETA)
            td_errors, loss = self.learn(batch, tau)
            self.memory.priority_update(inds, np.clip(np.abs(td_errors), PRIO_MIN, PRIO_MAX))
            losses.append(loss)

        return losses

    def explore(self, state): # exploration action
        action = []
        for i, s in enumerate(state):
            with torch.no_grad():
                action.append(self.ac_explorer.act(i, s).cpu().numpy())
        return np.clip(action, -1, +1)

    def exploit(self, state): # exploitation action
        action = []
        for s in state:
            with torch.no_grad():
                action.append(self.ac_target.act(-1, s).cpu().numpy())
        return np.clip(action, -1, +1)

    def _backprop(self, optim, loss, params):
        # learn
        optim.zero_grad() # scatter previous optimizer leftovers
        loss.backward() # propagate gradients
        torch.nn.utils.clip_grad_norm_(params, self.clip_norm) # avoid (inf, nan) stuffs
        optim.step() # backprop trigger

    def learn(self, batch, tau):
        """
        Order of learning : Critic first or Actor first
        - depends on who should learn Encoder layer
        - second in turn will learn also encoder, but need to be reflected in ac.py
        - aka *_parameters() should returns also encoder.parameters()
        - currently actor will learn encoder
        """
        states, actions, rewards, n_states = zip(*batch)
        states, actions, rewards, n_states = np.vstack(states), np.vstack(actions), np.vstack(rewards), np.vstack(n_states)
        # need to stack actions appropriatelly
        actions = actions.reshape(len(actions), self.n_rewards, -1)
        actions = np.concatenate([ actions[:, i, :] for i in range(self.n_rewards) ], 0)
        actions = torch.from_numpy(actions)

        losses = []

        # func approximators; self play
        with torch.no_grad():
            n_qa = self.ac_target(n_states)
        # TD(0) with k-step estimators
        td_targets = torch.tensor(rewards) + self.n_discount * n_qa
        for i in range(len(self.ac_explorer.critic)):
# learn ACTOR
            # func approximators; self play
            qa = self.ac_explorer(states, i)
            # DDPG + advantage learning with TD-learning
            td_error = qa - td_targets # w.r.t to self-played action from *0-step* state !!
            td_error = td_error.sum(1) # we try to maximize its sum/mean as cooperation matters now
            actor_loss = -td_error.mean()
            # learn!
            self._backprop(self.actor_optimizer, actor_loss, self.ac_explorer.actor_parameters())

# learn CRITIC
            # estimate reward
            q_replay = self.ac_explorer.value(states, actions, i)
            # calculate loss via TD-learning
            critic_loss = F.mse_loss(q_replay, td_targets)#F.smooth_l1_loss(q_replay, td_targets)#
            # learn!
            self._backprop(self.critic_optimizer[i], critic_loss, self.ac_explorer.critic_parameters(i))
            # propagate updates to target network ( network we trying to effectively learn )
            self._soft_update(self.ac_explorer.critic[i].parameters(), self.ac_target.critic[i], tau)

            losses.append(critic_loss.item())

        # propagate updates to target network ( network we trying to effectively learn )
        self._soft_update_mean(self.ac_explorer.actor, self.ac_target.actor[0], tau)

        for explorer in self.ac_explorer.actor:
            explorer.remove_noise() # lets go from zero ( noise ) !!

        return td_error.detach().cpu().numpy(), np.hstack([[actor_loss.item()], losses])

    def _soft_update_mean(self, explorers, targets, tau):
        if not tau:
            return

        params = np.mean([list(explorer.parameters()) for explorer in explorers], 0)
        self._soft_update(params, targets, tau)

    def _soft_update(self, explorer_params, targets, tau):
        if not tau:
            return

        for target_w, explorer_w in zip(targets.parameters(), explorer_params):
            target_w.data.copy_(
                target_w.data * (1. - tau) + explorer_w.data * tau)
