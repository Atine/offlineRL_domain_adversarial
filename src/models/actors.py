import torch
import torch.nn as nn
from torch import distributions as pyd


from src.models import weight_init


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = pyd.utils._standard_normal(
            shape, dtype=self.loc.dtype, device=self.loc.device
        )
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class ActorDefault(nn.Module):
    def __init__(
        self, repr_dim, action_dim, feature_dim, hidden_dim, pnorm=False
    ):
        super().__init__()

        self.log_std_min = -20
        self.log_std_max = 2

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim[0]),
        )

        self.apply(weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = TruncatedNormal(mu, std)
        return dist

    def save(self, fp):
        torch.save(self.state_dict(), fp)

    def load(self, fp, cpu=False):
        map_location = torch.device("cpu") if cpu else torch.device("cuda")
        self.load_state_dict(torch.load(fp, map_location))


class WrapperNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def sample(self, clip=None, sample_shape=torch.Size()):
        x_t = self.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        return action


class ActorSimple(nn.Module):
    def __init__(self, repr_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        self.log_std_min = -20
        self.log_std_max = 2

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.mu_head = nn.Linear(hidden_dim, action_dim[0])
        self.logstd_head = nn.Linear(hidden_dim, action_dim[0])

        self.apply(weight_init)

    def forward(self, obs, std=None):
        h = self.trunk(obs)
        h = self.policy(h)
        mu = self.mu_head(h)
        log_std = torch.tanh(self.logstd_head(h))

        log_std = torch.clamp(
            log_std, min=self.log_std_min, max=self.log_std_max
        )
        std = log_std.exp()

        dist = WrapperNormal(mu, std)
        return dist

    def save(self, fp):
        torch.save(self.state_dict(), fp)

    def load(self, fp, cpu=False):
        map_location = torch.device("cpu") if cpu else torch.device("cuda")
        self.load_state_dict(torch.load(fp, map_location))
