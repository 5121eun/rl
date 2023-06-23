import torch
from torch import nn

from torch.distributions import Categorical

class Reinforce:
    def __init__(self, env, n_acts: int, pi: nn.Module, pi_opt: torch.optim, 
                 n_batchs = 32, gamma = 0.98):
        super().__init__()
        
        self.env = env
        self.n_acts = n_acts
                
        self.pi = pi
        self.pi_opt = pi_opt
                
        self.gamma = gamma
        self.n_batchs = n_batchs     

        self.datas = []   
        
    def train(self, n_epis, print_interval=20):
        env = self.env
        score = 0.0
        step = 0

        for epi in range(n_epis):
            done = False
            s = env.reset()[0]
            
            while not done:
                a, a_prob = self.get_action(s)
                s_p, r, done, _, _ = env.step(a)
                d_mask = 0 if done else 1
                self.datas.append((a_prob, r/100, d_mask))
                env.render()

                s = s_p
                score += r
                step += 1
            
            self.update()
            
            if epi % print_interval == 0 and epi != 0:
                print(f"step: {step}, score: {score / print_interval}")
                score = 0

    def get_action(self, s):
        prob = self.pi(torch.tensor(s).float())
        m = Categorical(prob)
        a = m.sample().item()
        return a, prob[a]
            
    def update(self):
        g = 0

        self.pi_opt.zero_grad()
        for a_prob, r, d_mask in self.datas[::-1]:

            g = r + self.gamma * g * d_mask

            loss = - torch.log(a_prob) * g
            loss.backward()

        self.pi_opt.step()
        self.datas = []
        