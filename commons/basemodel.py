import torch
import numpy as np

class BaseModel():
    def __init__(self, max_len = 9999):

        self.datas = []
        self.max_len = max_len

    def put(self, data):
        if len(self.datas) > self.max_len:
            self.datas.pop(0)
        
        self.datas.append(data)

    def sample(self):
        lst = [[] for i in range(len(self.datas[0]))]
        for i in range(len(self.datas)):
            for j in range(len(self.datas[i])):
                d = self.datas[i][j]
                d = d if isinstance(d, np.ndarray) else np.array([d])
                lst[j].append(d)
        
        self.datas = []

        return [torch.tensor(np.array(d), dtype=torch.int64 if d[0].dtype == 'int' else torch.float) for d in lst]
            