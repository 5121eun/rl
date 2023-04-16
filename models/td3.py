import torch
from torch import nn
import torch.nn.functional as F

import copy

from commons.replaybuffer import ReplayBuffer

class TD3:
    def __init__(self, env, n_acts: int, act: nn.Module, act_opt: torch.optim, 
                 cri1: nn.Module, cri1_opt:torch.optim, 
                 cri2: nn.Module, cri2_opt:torch.optim, 
                 n_buffer=10000, n_batchs=32, d = 2, gamma = 0.98, tau = 5e-3, eps = 0.3, act_noise = 0.1, noise_range=(-0.5, 0.5), act_range=(-2, 2)):
        
        self.env = env
        self.n_acts = n_acts
                
        self.act = act
        self.act_opt = act_opt
        self.act_tg = copy.deepcopy(act)        
        
        self.cri1 = cri1
        self.cri1_opt = cri1_opt
        self.cri1_tg = copy.deepcopy(cri1)
        
        self.cri2 = cri2
        self.cri2_opt = cri2_opt
        self.cri2_tg = copy.deepcopy(cri2)

        self.buffer = ReplayBuffer(n_buffer)

        self.n_batchs = n_batchs
        
        self.gamma = gamma

        self.act_noise = act_noise
        self.noise_range = noise_range
        self.act_range = act_range
        
        self.d = d
        self.eps = eps
        self.tau = tau
                
    def train(self, n_epis, n_rollout, print_interval=200):
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
            
                if self.buffer.size() > self.n_batchs:
                    self.update_cri()
                    
                    if t % self.d == 0:
                        self.update_act()
            
                if t % print_interval == 0 and t != 0:
                    print(f"epi: {epi}, score: {score / print_interval}, n_buffer: {self.buffer.size()}")
                    score = 0.0
    
        
    def get_action(self, s):
        eps = torch.randn(self.n_acts) * self.act_noise
        a = torch.clamp(self.act(torch.from_numpy(s).float()) + eps, *self.act_range)
        return a.detach().numpy()
                
    def update_cri(self):
        s, a, r, s_p = self.buffer.sample(self.n_batchs)
            
        eps = torch.clamp(torch.randn(self.n_acts) *self.act_noise, *self.noise_range)
        a_hat = torch.clamp(self.act_tg(s_p) + eps, *self.act_range)

        y = r + self.gamma * torch.min(self.cri1_tg([s_p, a_hat]), self.cri2_tg([s_p, a_hat]))
            
        cri1_loss = F.mse_loss(self.cri1([s, a]), y.detach())
        cri2_loss = F.mse_loss(self.cri2([s, a]), y.detach())
                    
        self.cri1_opt.zero_grad()
        cri1_loss.backward()
        self.cri1_opt.step()
            
        self.cri2_opt.zero_grad()
        cri2_loss.backward()
        self.cri2_opt.step()
        
    def update_act(self):
        s = self.buffer.sample(self.n_batchs)[0]
        
        act_loss = - self.cri1([s, self.act(s)]).mean()
        self.act_opt.zero_grad()
        act_loss.backward()
        self.act_opt.step()
        
        self.update_tg(self.act_tg, self.act)
        self.update_tg(self.cri1_tg, self.cri1)
        self.update_tg(self.cri2_tg, self.cri2)
        
    def update_tg(self, target, source):
        for tg_params, sr_params in zip(target.parameters(), source.parameters()):
            tg_params.data.copy_(
                sr_params.data * self.tau + (1 - self.tau) * tg_params
            )