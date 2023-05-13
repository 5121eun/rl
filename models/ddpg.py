import torch
from torch import nn
import torch.nn.functional as F

import copy

from commons.replaybuffer import ReplayBuffer

class DDPG:
    def __init__(self, env, n_acts: int, act: nn.Module, act_opt: torch.optim, 
                 cri: nn.Module, cri_opt:torch.optim, 
                 n_buffer = 10000, n_batchs = 32, gamma = 0.98, act_noise = 0.1, act_range = (-1, 1), tau = 0.005):
        
        self.env = env
        self.n_acts = n_acts
                
        self.act = act
        self.act_tg = copy.deepcopy(act)
        self.act_opt = act_opt
        
        self.cri = cri
        self.cri_tg = copy.deepcopy(cri)
        self.cri_opt = cri_opt
        
        self.buffer = ReplayBuffer(n_buffer)

        self.n_batchs = n_batchs

        self.gamma = gamma
        self.tau = tau

        self.act_noise = act_noise
        self.act_range = act_range
        
    def train(self, n_epis, n_epochs, n_rollout, n_update=10, print_interval=20):
        env = self.env
        step = 0

        for epi in range(n_epis):
            s = env.reset()[0]
            score = 0.0
            
            for epoch in range(n_epochs):
                for t in range(n_rollout):
                    a = self.get_action(s)
                    s_p, r, _, _, _ = env.step(a)
                    self.buffer.put((s, a, r/100, s_p))
                    env.render()
                    
                    s = s_p
                    score += r
                    step += 1
                
                for n in range(n_update):
                    self.update()

                if epoch % print_interval == 0 and epoch != 0:
                    print(f"step: {step}, score: {score / print_interval}, n_buffer: {self.buffer.size()}")
                    score = 0.0
    
    def get_action(self, s):
        eps = torch.randn(self.n_acts) * self.act_noise
        a = torch.clamp(self.act(torch.from_numpy(s).float()) + eps, *self.act_range)
        return a.detach().numpy()
            
    def update(self):
        s, a, r, s_p = self.buffer.sample(self.n_batchs)
        
        y = r + self.gamma * self.cri_tg([s_p, self.act_tg(s_p)])
        cri_loss = F.mse_loss(self.cri([s, a]), y.detach())
        self.cri_opt.zero_grad()
        cri_loss.backward()
        self.cri_opt.step()
            
        act_loss = - self.cri([s, self.act(s)]).mean()
        self.act_opt.zero_grad()
        act_loss.backward()
        self.act_opt.step()
            
        self.update_tg(self.act_tg, self.act)
        self.update_tg(self.cri_tg, self.cri)
        
    def update_tg(self, target, source):
        for tg_params, sr_params in zip(target.parameters(), source.parameters()):
            tg_params.data.copy_(
                sr_params.data * self.tau + (1 - self.tau) * tg_params
            )