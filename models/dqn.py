import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import copy

from commons.replaybuffer import ReplayBuffer

class DQN:
    def __init__(self, env, n_acts: int, qnet: nn.Module, optim: torch.optim, n_batchs=32, buffer_limit=10000, gamma = 0.98, update_iter=10):
        
        self.env = env
        self.n_acts = n_acts
        
        self.q = qnet
        self.q_tg = copy.deepcopy(qnet)
        
        self.optim = optim

        self.buffer = ReplayBuffer(buffer_limit)

        self.update_iter = update_iter
        
        self.gamma = gamma
        self.n_batchs = n_batchs
        
        
    def train(self, n_epis, print_interval=20):
        env = self.env
        
        score = 0.0
        for epi in range(n_epis):
            eps = max(0.01, 0.08 - 0.01 * (epi/200))
            done = False
            s = env.reset()[0]
            
            while not done:
                a = self.get_action(eps, s)
                s_p, r, done, _, _ = env.step(a)
                d_mask = 0 if done else 1
                self.buffer.put((s, a, r/100, s_p, d_mask))
                
                env.render()
                
                s = s_p
                score += r
                
            if self.buffer.size() > self.n_batchs:
                self.update_policy()
            
            if epi % print_interval == 0 and epi != 0:
                self.q_target.load_state_dict(self.q.state_dict())
                print(f"epi: {epi}, score: {score}, n_buffer: {self.buffer.size()}, eps: {eps * 100: .1f}")

    def get_action(self, eps, s):
        randint = np.random.random()
        if randint < eps:
            return np.random.randint(0, self.n_acts, 1)[0]
        else: 
            return self.q(torch.from_numpy(s)).argmax().item()
            
    def update_policy(self): ##TODO: update iter 바꿀지 말지
        for i in range(self.update_iter):
            s, a, r, s_p, d_mask = self.buffer.sample(self.n_batchs)
            max_q_prime = self.q_tg(s_p).max(dim=-1)[0].unsqueeze(1)
            
            y = r + (self.gamma * max_q_prime * d_mask)
            loss = F.mse_loss(self.q(s).gather(1, a), y)        

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
    


