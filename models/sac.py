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
                 n_buffer=10000, n_batchs=32, gamma = 0.99, alpha = 0.1, eps = 0.3, tau = 5e-3):
        
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
        self.target_entorpy = - 1.0
                
    def train(self, n_epis, n_rollout, print_interval=20):
        env = self.env
        
        score = 0.0
        for epi in range(n_epis):
            s = env.reset()[0]

            for t in range(n_rollout):
                a = self.get_action(s)
                s_p, r, _, _, _ = env.step(a)
                self.buffer.put((s, a, r/100, s_p))
                env.render()

                s = s_p
                score += r
                                        
                if t > self.n_batchs:
                    self.update()
            if epi % print_interval == 0 and epi != 0:
                print(f"epi: {epi}, score: {score / print_interval}, n_buffer: {self.buffer.size()}")
                score = 0.0
            
    def update(self):
        s, a, r, s_p = self.buffer.sample(self.n_batchs)
        
        cur_a_p, cur_a_p_log_prob = self.act(s_p)
        entorpy = - self.log_alpha.exp() * cur_a_p_log_prob
        q_min = torch.min(self.qcri1_tg([s_p, cur_a_p]), self.qcri2_tg([s_p, cur_a_p]))
        y = r + self.gamma * (q_min + entorpy) 
        
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

        log_alpha_ls = - (self.log_alpha.exp() * (cur_a_log_prob + self.target_entorpy).detach()).mean()
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