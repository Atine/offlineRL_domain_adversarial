import torch
import torch.nn as nn
import torch.nn.functional as F


class Value(nn.Module):
    def __init__(self, repr_dim):
        super().__init__()

        # V architecture
        self.l1 = nn.Linear(repr_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, obs):
        v = F.relu(self.l1(obs))
        v = F.relu(self.l2(v))
        v = self.l3(v)
        return v

    def save(self, fp):
        torch.save(self.state_dict(), fp)

    def load(self, fp, cpu=False):
        map_location = torch.device("cpu") if cpu else torch.device("cuda")
        self.load_state_dict(torch.load(fp, map_location))
