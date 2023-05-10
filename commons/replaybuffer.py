import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, limit):
        self.datas = []
        self.limit = limit
        
    def put(self, data):
        if len(self.datas) + 1 == self.limit:
            self.datas.pop(0)
        self.datas.append(data)
        
    def size(self):
        return len(self.datas)
        
    def sample(self, n: int):
        idx = np.random.randint(0, len(self.datas) - 1, min(len(self.datas), n))

        lst = [[] for i in range(len(self.datas[0]))]
        for i in idx:
            for j in range(len(self.datas[i])):
                d = self.datas[i][j]
                d = d if isinstance(d, np.ndarray) else np.array([d])
                lst[j].append(d)
            
        return [torch.tensor(np.array(d), dtype=torch.int64 if d[0].dtype == 'int' else torch.float) for d in lst]