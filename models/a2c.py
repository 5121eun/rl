from torch.distributions import Categorical
import torch.nn.functional as F

import torch
import torch.nn as nn

from commons.basemodel import BaseModel


class A2C(BaseModel):
    def __init__(self, env, n_acts: int, act: nn.Module, act_opt: torch.optim, 
                 cri: nn.Module, cri_opt:torch.optim, gamma = 0.98):
        super().__init__()
        
        self.env = env
        self.n_acts = n_acts
                
        self.act = act
        self.act_opt = act_opt
        
        self.cri = cri
        self.cri_opt = cri_opt
                        
        self.gamma = gamma
        
    def train(self, n_epis, n_rollout, n_update, print_interval=20):
        env = self.env
        score = 0.0
        step = 0

        for epi in range(n_epis):
            done = False
            s = env.reset()[0]

            while not done:
                for t in range(n_rollout):
                    a = self.get_action(s)
                    s_p, r, done, _, _ = env.step(a)
                    d_mask = 0 if done else 1
                    self.put((s, a, r/100, s_p, d_mask))
                    env.render()

                    s = s_p
                    score += r
                    step += 1

                    if done:
                        break

                n_batch = n_rollout//n_update
                samples = self.sample()

                for i in range(0, n_rollout, n_batch):
                    sample = [s[i:i+n_batch] for s in samples]
                    self.update(sample)
            
            if epi % print_interval == 0 and epi != 0:
                print(f"step: {step}, score: {score / print_interval}")
                score = 0

    def get_action(self, s):
        prob = self.act(torch.tensor(s).float())
        m = Categorical(prob)
        return m.sample().item()
            
    def update(self, sample):
        s, a, r, s_p, d_mask = sample
        
        td_target = r + self.gamma * self.cri(s_p) * d_mask
        adv = td_target - self.cri(s)
            
        pi = self.act(s)
        pi_a = pi.gather(1, a)
        act_loss = - torch.log(pi_a) * adv.detach()
        act_loss = act_loss.mean()
        self.act_opt.zero_grad()
        act_loss.backward()
        self.act_opt.step()
            
        cri_loss = F.mse_loss(self.cri(s), td_target.detach())
        self.cri_opt.zero_grad()
        cri_loss.backward()
        self.cri_opt.step()