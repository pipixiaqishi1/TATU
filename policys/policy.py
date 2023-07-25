import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy


class SACPolicy(nn.Module):
    def __init__(
        self, 
        actor, 
        critic1, 
        critic2,
        actor_optim, 
        critic1_optim, 
        critic2_optim,
        action_space,
        dist, 
        tau=0.005, 
        gamma=0.99, 
        alpha=0.2,
        device="cpu"
    ):
        super().__init__()

        self.actor = actor
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()

        self.actor_optim = actor_optim
        self.critic1_optim = critic1_optim
        self.critic2_optim = critic2_optim

        self.action_space = action_space
        self.dist = dist

        self._tau = tau
        self._gamma = gamma

        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self._alpha_optim = alpha
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha
        
        self.__eps = np.finfo(np.float32).eps.item()

        self._device = device
    
    def train(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
    
    def _sync_weight(self):
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
    
    def forward(self, obs, deterministic=False):
        dist = self.actor.get_dist(obs)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action)

        action_scale = torch.tensor((self.action_space.high - self.action_space.low) / 2, device=action.device)
        squashed_action = torch.tanh(action)
        log_prob = log_prob - torch.log(action_scale * (1 - squashed_action.pow(2)) + self.__eps).sum(-1, keepdim=True)

        return squashed_action, log_prob

    def sample_action(self, obs, deterministic=False):
        action, _ = self(obs, deterministic)
        return action.cpu().detach().numpy()

    def learn(self, data):
        obs, actions, next_obs, terminals, rewards = data["observations"], \
            data["actions"], data["next_observations"], data["terminals"], data["rewards"]
        
        rewards = torch.as_tensor(rewards).to(self._device)
        terminals = torch.as_tensor(terminals).to(self._device)
        
        # update critic
        q1, q2 = self.critic1(obs, actions).flatten(), self.critic2(obs, actions).flatten()
        with torch.no_grad():
            next_actions, next_log_probs = self(next_obs)
            next_q = torch.min(
                self.critic1_old(next_obs, next_actions), self.critic2_old(next_obs, next_actions)
            ) - self._alpha * next_log_probs
            target_q = rewards.flatten() + self._gamma * (1 - terminals.flatten()) * next_q.flatten()
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()
        critic2_loss = ((q2 - target_q).pow(2)).mean()
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor
        a, log_probs = self(obs)
        q1a, q2a = self.critic1(obs, a).flatten(), self.critic2(obs, a).flatten()
        actor_loss = (self._alpha * log_probs.flatten() - torch.min(q1a, q2a)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        self._sync_weight()

        result =  {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item()
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()
        
        return result
        
class CQLPolicy(nn.Module):
    def __init__(
        self, 
        actor, 
        critic1, 
        critic2,
        actor_optim, 
        critic1_optim, 
        critic2_optim,
        action_space,
        dist, 
        tau=0.005, 
        gamma=0.99, 
        alpha=0.2,
        device="cpu",
        min_q_version=3,
        min_q_weight=5.0,
        lagrange_thresh=5,
    ):
        super().__init__()

        self.actor = actor
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()

        self.actor_optim = actor_optim
        self.critic1_optim = critic1_optim
        self.critic2_optim = critic2_optim

        self.action_space = action_space
        self.dist = dist

        self.critic_criterion = nn.MSELoss()
        self.min_q_version = min_q_version
        self.min_q_weight = min_q_weight

        self._tau = tau
        self._gamma = gamma
        
        if 0:
            self.lagrange_thresh = lagrange_thresh
        else:
            self.lagrange_thresh = -1
            
        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self._alpha_optim = alpha
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha
        
        self.__eps = np.finfo(np.float32).eps.item()

        self._device = device
    
    def train(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
    
    def _sync_weight(self):
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
    
    def forward(self, obs, deterministic=False):
        dist = self.actor.get_dist(obs)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action)

        action_scale = torch.tensor((self.action_space.high - self.action_space.low) / 2, device=action.device)
        squashed_action = torch.tanh(action)
        log_prob = log_prob - torch.log(action_scale * (1 - squashed_action.pow(2)) + self.__eps).sum(-1, keepdim=True)

        return squashed_action, log_prob

    def sample_action(self, obs, deterministic=False):
        action, _ = self(obs, deterministic)
        return action.cpu().detach().numpy()
    
    def _get_tensor_values(self, obs, actions, network):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int (action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        preds = network(obs_temp, actions)
        preds = preds.view(obs.shape[0], num_repeat, 1)

        return preds

    def _get_policy_actions(self, obs, num_actions, network=None):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        new_obs_actions,new_obs_log_pi= network(obs_temp, deterministic=False)

        return new_obs_actions, new_obs_log_pi.view(obs.shape[0], num_actions, 1)

    def learn(self, data):
        obs, actions, next_obs, terminals, rewards = data["observations"], \
            data["actions"], data["next_observations"], data["terminals"], data["rewards"]

        obs = torch.as_tensor(obs).to(self._device)
        actions = torch.as_tensor(actions).to(self._device)
        next_obs = torch.as_tensor(next_obs).to(self._device)
        rewards = torch.as_tensor(rewards).to(self._device)
        terminals = torch.as_tensor(terminals).to(self._device)

        # update critic
        q1_pred, q2_pred = self.critic1(obs, actions).flatten(), self.critic2(obs, actions).flatten()
        with torch.no_grad():
            next_actions, next_log_probs = self(next_obs)
            next_q = torch.min(
                self.critic1_old(next_obs, next_actions), self.critic2_old(next_obs, next_actions)
            ) - self._alpha * next_log_probs
            target_q = rewards.flatten() + self._gamma * (1 - terminals.flatten()) * next_q.flatten()
        
        qf1_loss = self.critic_criterion(q1_pred, target_q)
        qf2_loss = self.critic_criterion(q2_pred, target_q)
        
        ## add CQL
        random_actions_tensor = torch.FloatTensor(q2_pred.shape[0] * 10, actions.shape[-1]).uniform_(-1, 1).to(self._device)
        curr_actions_tensor, curr_log_pis = self._get_policy_actions(obs, num_actions=10, network=self.forward)
        new_curr_actions_tensor, new_log_pis = self._get_policy_actions(next_obs, num_actions=10, network=self.forward)
        q1_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.critic1)
        q2_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.critic2)
        q1_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.critic1)
        q2_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.critic2)
        q1_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.critic1)
        q2_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.critic2)

        # print(q1_rand.shape, q2_pred.shape, q1_next_actions.shape, q1_curr_actions.shape)
        cat_q1 = torch.cat([q1_rand, q1_pred.view(-1,1,1), q1_next_actions, q1_curr_actions], 1)
        cat_q2 = torch.cat([q2_rand, q2_pred.view(-1,1,1), q2_next_actions, q2_curr_actions], 1)

        if self.min_q_version == 3:
            # importance sammpled version
            random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
            cat_q1 = torch.cat(
                [q1_rand - random_density, q1_next_actions - new_log_pis.detach(), q1_curr_actions - curr_log_pis.detach()], 1
            )
            cat_q2 = torch.cat(
                [q2_rand - random_density, q2_next_actions - new_log_pis.detach(), q2_curr_actions - curr_log_pis.detach()], 1
            )
            
        min_qf1_loss = torch.logsumexp(cat_q1, dim=1,).mean() * self.min_q_weight
        min_qf2_loss = torch.logsumexp(cat_q2, dim=1,).mean() * self.min_q_weight
                    
        """Subtract the log likelihood of data"""
        min_qf1_loss = min_qf1_loss - q1_pred.mean() * self.min_q_weight
        min_qf2_loss = min_qf2_loss - q2_pred.mean() * self.min_q_weight
        
        
        if self.lagrange_thresh >= 0:
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
            min_qf1_loss = alpha_prime * (min_qf1_loss - self.lagrange_thresh)
            min_qf2_loss = alpha_prime * (min_qf2_loss - self.lagrange_thresh)

            self.alpha_prime_opt.zero_grad()
            alpha_prime_loss = (-min_qf1_loss - min_qf2_loss)*0.5 
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_opt.step()

        qf1_loss = qf1_loss + min_qf1_loss
        qf2_loss = qf2_loss + min_qf2_loss

        self.critic1_optim.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        qf2_loss.backward(retain_graph=True)
        self.critic2_optim.step()

        # update actor
        a, log_probs = self(obs)
        q1a, q2a = self.critic1(obs, a).flatten(), self.critic2(obs, a).flatten()
        actor_loss = (self._alpha * log_probs.flatten() - torch.min(q1a, q2a)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        self._sync_weight()

        result =  {
            "loss/actor": actor_loss.item(),
            "loss/critic1": qf1_loss.item(),
            "loss/critic2": qf2_loss.item()
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()
        
        return result