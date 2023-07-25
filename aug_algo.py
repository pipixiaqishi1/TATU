import torch
from torch.nn import Module
import torch.nn as nn
import numpy as np
from copy import deepcopy
from loguru import logger
from tianshou.data import Batch

from offlinerl.algo.base import BaseAlgo

from offlinerl.utils.exp import setup_seed

from offlinerl.utils.data import ModelBuffer
from offlinerl.utils.net.model.ensemble import EnsembleTransition

from torch.autograd import Variable
import os

import torch.nn.functional as F

# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device


    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability 
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        
        u = self.decode(state, z)

        return u, mean, std


    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

def algo_init(args):
    logger.info('Run algo_init function')

    setup_seed(args["seed"])
    if args['task']:
        obs_dim, action_dim = np.prod(args["obs_shape"]), args["action_dim"]
        args["obs_dim"], args["action_dim"] = obs_dim, action_dim
        max_action, min_action = args['max_action'], args['min_action']
    else:
        raise NotImplementedError
    
    forward_transition = EnsembleTransition(obs_dim, action_dim, 200, 4, args['n_ensembles']).to(args['device'])
    forward_transition_optim = torch.optim.Adam(forward_transition.parameters(), lr=0.001, weight_decay=0.000075)

    ## forward variational auto-encoder
    latent_dim = action_dim * 2
    fvae = VAE(obs_dim, action_dim, latent_dim, max_action, args['device']).to(args['device'])
    fvae_optim = torch.optim.Adam(fvae.parameters())

    return {
        "forward_transition" : {"net" : forward_transition, "opt" : forward_transition_optim},
        "fvae" : {"net" : fvae, "opt" : fvae_optim},
    }

class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args, logger):
        super(AlgoTrainer, self).__init__(args)
        self.args = args
        self.logger = logger
        self.forward_transition = algo_init['forward_transition']['net']
        self.forward_transition_optim = algo_init['forward_transition']['opt']
        self.forward_transition_optim_secheduler = torch.optim.lr_scheduler.ExponentialLR(self.forward_transition_optim, gamma=0.99)
        self.selected_transitions = None

        self.fvae = algo_init['fvae']['net']
        self.fvae_optim = algo_init['fvae']['opt']

        self.device = args['device']
        self.forward_transition_train_epoch = self.args['forward_transition_train_epoch']
        self._best_loss = {i: (None, 1e10) for i in range(self.forward_transition.ensemble_size)}
        self._max_epochs_since_update = 5
        self._max_disc = None
        self._pessimism_coef = args['pessimism_coef']

    def train(self, train_buffer,model_buffer):

        self.train_transition(train_buffer) ## train model
        self.train_vae(train_buffer)  ## train VAE
        self.forward_transition.requires_grad_(False)
        batch = self.rollout_transition(train_buffer,model_buffer,
                                   self.forward_transition, self.fvae)
        return batch
    
    def get_policy(self):
        pass

    def _stop_train_model(self,epoch,loss):
        updated = False
        for i in range(self.forward_transition.ensemble_size):
            current = loss[i]
            _, best = self._best_loss[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._best_loss[i] = (epoch, current)
                updated = True
                # print('epoch {} | updated {} | improvement: {:.4f} | best: {:.4f} | current: {:.4f}'.format(epoch, i, improvement, best, current))
        
        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1

        if self._epochs_since_update > self._max_epochs_since_update:
            # print('[ BNN ] Breaking at epoch {}: {} epochs since update ({} max)'.format(epoch, self._epochs_since_update, self._max_epochs_since_update))
            return True
        else:
            return False
        
    def train_transition(self, buffer):
        data_size = buffer.size
        val_size = min(int(data_size * 0.2) + 1, 1000)
        train_size = data_size - val_size
        train_splits, val_splits = torch.utils.data.random_split(range(data_size), (train_size, val_size))
        all_data = buffer.sample_all()

        train_data = {}
        train_data['observations'] = all_data['observations'][train_splits.indices]
        train_data['actions'] = all_data['actions'][train_splits.indices]
        train_data['next_observations'] = all_data['next_observations'][train_splits.indices]
        train_data['terminals'] = all_data['terminals'][train_splits.indices]
        train_data['rewards'] = all_data['rewards'][train_splits.indices]
        valdata = {}
        valdata['observations'] = all_data['observations'][val_splits.indices]
        valdata['actions'] = all_data['actions'][val_splits.indices]
        valdata['next_observations'] = all_data['next_observations'][val_splits.indices]
        valdata['terminals'] = all_data['terminals'][val_splits.indices]
        valdata['rewards'] = all_data['rewards'][val_splits.indices]

        batch_size = self.args['batch_size']

        forward_val_losses = [float('inf') for i in range(self.forward_transition.ensemble_size)]

        epoch = 0
        forward_cnt = 0
        break_train = False
        ## train forward transition
        while True:
            epoch += 1
            idxs = np.random.randint(train_data['observations'].shape[0], size=[self.forward_transition.ensemble_size, train_data['observations'].shape[0]])
            for batch_num in range(int(np.ceil(idxs.shape[-1] / batch_size))):
                batch_idxs = idxs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
                batch = {}
                batch['actions'] = all_data['actions'][batch_idxs]
                batch['next_observations'] = all_data['next_observations'][batch_idxs]
                batch['terminals'] = all_data['terminals'][batch_idxs]
                batch['rewards'] = all_data['rewards'][batch_idxs]
                batch['observations'] = all_data['observations'][batch_idxs]
                self._train_forward_transition(self.forward_transition, batch, self.forward_transition_optim)
            forward_new_val_losses = self._eval_forward_transition(self.forward_transition, valdata)
            print('Forward validation loss: ', forward_new_val_losses)

            forward_indexes = []
            for i, new_loss, old_loss in zip(range(len(forward_val_losses)), forward_new_val_losses, forward_val_losses):
                if new_loss < old_loss:
                    forward_indexes.append(i)
                    forward_val_losses[i] = new_loss

            if len(forward_indexes) > 0:
                self.forward_transition.update_save(forward_indexes)
                forward_cnt = 0
            else:
                forward_cnt += 1

            break_train = self._stop_train_model(epoch,forward_val_losses)

            if break_train:
                break
            # if epoch >= self.forward_transition_train_epoch:
            if epoch >= self.forward_transition_train_epoch:
                break

            self.forward_transition_optim_secheduler.step()
        
        forward_indexes = self._select_best_indexes(forward_val_losses, n=self.args['n_elites'])
        self.forward_transition.set_select(forward_indexes)
        
        self.logger.print('Epoch:{}  val_losses:{} forward_indexes:{}'.format(epoch,forward_new_val_losses,forward_indexes))  
        return self.forward_transition
    
    def train_vae(self, buffer):
        for i in range(self.args['vae_train_epoch']):
            batch = buffer.sample(self.args['batch_size'])
            self._train_forward_vae(batch, self.fvae, self.fvae_optim)

    def compute_max_disc(self,forward_transition,train_buffer):
        all_offline_data = train_buffer.sample_all()

        observations = all_offline_data["observations"]
        actions = all_offline_data["actions"]
        max_disc = -100
        mini_batch = 1000
        slice_num = len(observations)// mini_batch
        alone_num = len(observations)% mini_batch
        for i in range(slice_num):
            obs = observations[i*mini_batch:(i+1)*mini_batch,:]
            act = actions[i*mini_batch:(i+1)*mini_batch,:]
            obs = torch.tensor(obs, device=self.device)
            act = torch.tensor(act, device=self.device)
            obs_action = torch.cat([obs, act], dim=-1)
            mini_max_disc = np.max(self.compute_disc(forward_transition,obs_action))
            if mini_max_disc > max_disc:
                max_disc = mini_max_disc
        if alone_num != 0:
            obs = observations[slice_num*mini_batch:,:]
            act = actions[slice_num*mini_batch:,:]
            obs = torch.tensor(obs, device=self.device)
            act = torch.tensor(act, device=self.device)
            obs_action = torch.cat([obs, act], dim=-1)
            mini_max_disc = np.max(self.compute_disc(forward_transition,obs_action))
            if mini_max_disc > max_disc:
                max_disc = mini_max_disc

        return max_disc

    def compute_disc(self,forward_transition,obs_action):
        next_obs_dists = forward_transition(obs_action)
        next_obses = next_obs_dists.sample()
        rewards = next_obses[:, :, -1:]
        next_obses = next_obses[:, :, :-1]

        next_obses_mode = next_obs_dists.mean[:, :, :-1]
        next_obs_mean = torch.mean(next_obses_mode, dim=0)
        diff = next_obses_mode - next_obs_mean
        disc = torch.max(torch.norm(diff, dim=-1), dim=0)[0].cpu().numpy()
        return disc


    def rollout_transition(self, train_buffer, model_buffer, forward_transition, fvae):
        for epoch in range(self.args['rollout_epoch']):
            # collect data
            with torch.no_grad():
                ## forward imagination
                obs = train_buffer.sample(int(self.args['rollout_batch_size']))['observations']
                obs = torch.tensor(obs, device=self.device)

                cumul_error = np.zeros(len(obs))
                if self._max_disc == None:
                    self._max_disc = self.compute_max_disc(forward_transition,train_buffer) 

                threshold = 1/self._pessimism_coef *self._max_disc
                halt_info ={"max_disc":self._max_disc,
                    "threshold":threshold,
                    "halt_num":[],
                    "halt_ratio":[],
                    "stop_ratio":[]} 

                for t in range(self.args['rollout_length']):
                    # sample from variational auto-encoder
                    action = self.fvae.decode(obs)
                    obs_action = torch.cat([obs, action], dim=-1)
                    next_obs_dists = forward_transition(obs_action)
                    next_obses = next_obs_dists.sample()
                    rewards = next_obses[:, :, -1:]
                    next_obses = next_obses[:, :, :-1]

                    next_obses_mode = next_obs_dists.mean[:, :, :-1]
                    next_obs_mean = torch.mean(next_obses_mode, dim=0)
                    diff = next_obses_mode - next_obs_mean
                    disagreement_uncertainty = torch.max(torch.norm(diff, dim=-1, keepdim=True), dim=0)[0]
                    # aleatoric_uncertainty = torch.max(torch.norm(next_obs_dists.stddev, dim=-1, keepdim=True), dim=0)[0]
                    uncertainty = disagreement_uncertainty # if self.args['uncertainty_mode'] == 'disagreement' else aleatoric_uncertainty

                    model_indexes = np.random.randint(0, next_obses.shape[0], size=(obs.shape[0]))
                    next_obs = next_obses[model_indexes, np.arange(obs.shape[0])]


                    reward = rewards[model_indexes, np.arange(obs.shape[0])]
                    reward -= self.args['reward_penalty_coef'] * uncertainty

                    disc = uncertainty.view(-1).cpu().numpy()
                    cumul_error += disc
                    known = np.where(cumul_error < threshold)[0]

                    
                    dones = torch.zeros_like(reward)

                    ## select high confidence states
                    conf_obs = obs.cpu().numpy()[known,:]
                    conf_action = action.cpu().numpy()[known,:]
                    conf_rew = reward.cpu().numpy()[known,:]
                    conf_dones = dones.cpu().numpy()[known,:]
                    conf_next_obs = next_obs.cpu().numpy()[known,:]

                    halt_info['halt_num'] = len(obs)-len(conf_obs)
                    halt_info['halt_ratio'] = 1 - len(conf_obs)/len(obs)
                    
                    model_buffer.add_batch(conf_obs,conf_next_obs,conf_action,conf_rew,conf_dones)

                    obs = next_obs

                self.logger.print('forward average reward: {}'.format(reward.mean().item()))
                self.logger.print('halt_info: {}'.format(halt_info))
        return

    def _select_best_indexes(self, metrics, n):
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        selected_indexes = [pairs[i][1] for i in range(n)]
        return selected_indexes

    def _train_forward_transition(self, transition, data, optim):
        dist = transition(torch.from_numpy(np.concatenate([data['observations'], data['actions']], axis=-1)).to(self.device))
        loss = - dist.log_prob(torch.from_numpy(np.concatenate([data['next_observations'], data['rewards']], axis=-1)).to(self.device))
        loss = loss.mean()

        loss = loss + 0.01 * transition.max_logstd.mean() - 0.01 * transition.min_logstd.mean()

        optim.zero_grad()
        loss.backward()
        optim.step()
        
    def _eval_forward_transition(self, transition, valdata):
        with torch.no_grad():
            dist = transition(torch.from_numpy(np.concatenate([valdata['observations'], valdata['actions']], axis=-1)).to(self.device))
            loss = ((dist.mean - torch.from_numpy(np.concatenate([valdata['next_observations'], valdata['rewards']], axis=-1)).to(self.device)) ** 2).mean(dim=(1,2))
            return list(loss.cpu().numpy())
        
    
    def _train_forward_vae(self, data, fvae, fvae_optim):
        state = torch.from_numpy(data['observations']).to(self.device)
        action = torch.from_numpy(data['actions']).to(self.device)
        # Variational Auto-Encoder Training
        recon, mean, std = fvae(state, action)
        recon_loss = F.mse_loss(recon, action)
        KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        fvae_loss = recon_loss + 0.5 * KL_loss

        fvae_optim.zero_grad()
        fvae_loss.backward()
        fvae_optim.step()
    