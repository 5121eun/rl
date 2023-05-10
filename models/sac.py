import torch
from torch import nn
import torch.nn.functional as F

import copy
import numpy as np

from commons.replaybuffer import ReplayBuffer

class SAC:
    def __init__(self, env, nactions: int, act: nn.Module, act_opt: torch.optim, 
                 qcri1: nn.Module, qcri1_opt:torch.optim, 
                 qcri2: nn.Module, qcri2_opt:torch.optim, 
                 n_buffer=10000, n_batchs=32, gamma = 0.99, alpha = 0.1, eps = 0.3, tau = 5e-3, target_entropy=None):
        
        self.env = env
        self.nactions = nactions
                
        self.act = act
        self.act_opt = act_opt
        
        
        self.qcri1 = qcri1
        self.qcri1_tg = copy.deepcopy(qcri1)
        self.qcri1_opt = qcri1_opt
        
        self.qcri2 = qcri2
        self.qcri2_tg = copy.deepcopy(qcri2)
        self.qcri2_opt = qcri2_opt

        self.buffer = ReplayBuffer(n_buffer)

        self.n_batchs = n_batchs

        
        self.gamma = gamma
        self.alpha = alpha
        
        self.eps = eps
        self.tau = tau

        self.log_alpha = torch.tensor(np.log(0.01))
        self.log_alpha.requires_grad = True
        self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=0.001)
        
        if target_entropy is None:
            self.target_entropy = - nactions
        else:
            self.target_entropy = target_entropy
                
    def train(self, n_epis, n_epochs, n_rollout, n_update=20, print_interval=20):
        env = self.env
        
        for epi in range(n_epis):
            s = env.reset()[0]
            done = False
            score = 0.0

            for epoch in range(n_epochs):
                for t in range(n_rollout):
                    a = self.get_action(s)
                    s_p, r, done, _, _ = env.step(a)
                    d_mask = 0.0 if done else 1.0
                    self.buffer.put((s, a, r/100, d_mask, s_p))
                    env.render()

                    s = s_p
                    score += r

                    if done:
                        break
                                            
                for t in range(n_update):
                    self.update()

                if epoch % print_interval == 0 and epoch != 0:
                    print(f"epoch: {epoch}, score: {score / print_interval}, n_buffer: {self.buffer.size()}")
                    score = 0.0
            
    def update(self):
        s, a, r, d_mask, s_p = self.buffer.sample(self.n_batchs)
        
        cur_a_p, cur_a_p_log_prob = self.act(s_p)
        entorpy = - self.log_alpha.exp() * cur_a_p_log_prob
        q_min = torch.min(self.qcri1_tg([s_p, cur_a_p]), self.qcri2_tg([s_p, cur_a_p]))
        y = r + self.gamma * (q_min + entorpy) * d_mask
        
        test=self.qcri1([s, a])
        qcri_ls1 = F.smooth_l1_loss(self.qcri1([s, a]), y.detach())

        self.qcri1_opt.zero_grad()
        qcri_ls1.backward()
        self.qcri1_opt.step()
        
        qcri_ls2 = F.smooth_l1_loss(self.qcri2([s, a]), y.detach())
        self.qcri2_opt.zero_grad()
        qcri_ls2.backward()
        self.qcri2_opt.step()
        
        cur_a, cur_a_log_prob = self.act(s)
        entorpy = - self.log_alpha.exp() * cur_a_log_prob
        q_min = torch.min(self.qcri1([s, cur_a]), self.qcri2([s, cur_a]))
        act_ls = - (q_min + entorpy).mean()
        self.act_opt.zero_grad()
        act_ls.backward()
        self.act_opt.step()

        log_alpha_ls = - (self.log_alpha.exp() * (cur_a_log_prob + self.target_entropy).detach()).mean()
        self.log_alpha_opt.zero_grad()
        log_alpha_ls.backward()
        self.log_alpha_opt.step()
            
        self.update_tg(self.qcri1_tg, self.qcri1)
        self.update_tg(self.qcri2_tg, self.qcri2)

 
    def get_action(self, s):
        a, _ = self.act(torch.from_numpy(s).float())
        return a.detach().numpy()
        
    def update_tg(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                source_param.data * self.tau + (1 - self.tau) * target_param
            )