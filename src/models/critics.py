import torch
import torch.nn as nn
import torch.nn.functional as F


from src.models import weight_init


class Critic(nn.Module):
    def __init__(
        self,
        repr_dim,
        action_dim,
        feature_dim,
        hidden_dim,
        dropout=False,
    ):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )

        self.l1 = nn.Linear(feature_dim + action_dim[0], hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        self.l4 = nn.Linear(feature_dim + action_dim[0], hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

        if dropout:
            self.drop1 = nn.Dropout()
            self.drop2 = nn.Dropout()
        else:
            self.drop1 = nn.Identity()
            self.drop2 = nn.Identity()

        self.apply(weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)

        q1 = F.relu(self.drop1(self.l1(h_action)))
        q1 = F.relu(self.drop2(self.l2(q1)))
        q1 = self.l3(q1)
        q2 = F.relu(self.drop1(self.l4(h_action)))
        q2 = F.relu(self.drop2(self.l5(q2)))
        q2 = self.l6(q2)

        return q1, q2

    def forward_to_last_hidden(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)

        q1 = F.relu(self.l1(h_action))
        q1 = F.relu(self.l2(q1))
        q2 = F.relu(self.l4(h_action))
        q2 = F.relu(self.l5(q2))
        return q1, q2

    def save(self, fp):
        torch.save(self.state_dict(), fp)

    def load(self, fp, cpu=False):
        map_location = torch.device("cpu") if cpu else torch.device("cuda")
        self.load_state_dict(torch.load(fp, map_location))
