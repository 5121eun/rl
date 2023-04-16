import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

from commons.basemodel import BaseModel

class PPO(BaseModel):
    def __init__(self, env, n_acts: int, act: nn.Module, act_opt: torch.optim, 
                 cri: nn.Module, cri_opt:torch.optim, 
                 n_batchs=32, gamma = 0.98, gae=0.95, eps=0.2):
        super().__init__()

        self.env = env
        self.n_acts = n_acts
        
        self.act = act
        self.act_opt = act_opt
        
        self.cri = cri
        self.cri_opt = cri_opt

        self.n_batchs = n_batchs
        
        self.gamma = gamma
        self.gae = gae
        self.eps = eps

        
        
    def train(self, n_epis, n_rollout, print_interval=20):
        env = self.env
        score = 0.0

        for epi in range(n_epis):
            done = False
            s = env.reset()[0]
            
            while not done:
                for t in range(n_rollout):
                    a, a_prob = self.get_action(s)
                    s_p, r, done, _, _ = env.step(a)
                    d_mask = 0 if done else 1
                    self.put((s, a, r/100, s_p, d_mask, a_prob))
                    env.render()

                    s = s_p
                    score += r

                    if done:
                        break
            
                self.update()
            
            if epi % print_interval == 0 and epi != 0:
                print(f"epi: {epi}, score: {score / print_interval}")
                score = 0

    def get_action(self, s):
        prob = self.act(torch.from_numpy(s).type(torch.float))
        m = Categorical(prob)
        a = m.sample().item()
        return a, prob[a].item()

    def update(self):
        s, a, r, s_p, d_mask, a_prob = self.sample()

        td_target = r + self.gamma * self.cri(s_p) * d_mask
        advs = (td_target - self.cri(s)).view(-1)

        a_hats = []
        a_hat = 0
        for adv in advs.detach().numpy()[::-1]:
            a_hat = adv + self.gamma * self.gae * a_hat
            a_hats.append(a_hat)
        
        a_hats.reverse()
        a_hats = torch.tensor(a_hats)
        
        cri_loss = F.mse_loss(self.cri(s), td_target.detach())
        self.cri_opt.zero_grad()
        cri_loss.backward()
        self.cri_opt.step()
        
        pi = self.act(s)
        pi_a = pi.gather(1, a) / a_prob
        
        act_loss = - torch.min(pi_a * a_hats, torch.clamp(pi_a, 1 - self.eps, 1 + self.eps) * a_hats)
        act_loss = act_loss.mean()
        self.act_opt.zero_grad()
        act_loss.backward()
        self.act_opt.step()
