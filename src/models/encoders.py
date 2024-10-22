import torch
import torch.nn as nn


from src.models import weight_init, DropBlock2D


class GradientReverse(torch.autograd.Function):
    scale = 1.0

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReverse.scale * grad_output


def grad_reverse(x, scale=1.0):
    GradientReverse.scale = scale
    return GradientReverse.apply(x)


class Encoder(nn.Module):
    """encoder with norev branch return"""

    def __init__(
        self,
        obs_dim=(9, 84, 84),
        feat_dim=32,
        dropblock=False,
        act=nn.ReLU,
    ):
        super().__init__()

        assert len(obs_dim) == 3
        self.repr_dim = feat_dim * 35 * 35
        self.act = act()

        self.conv1 = nn.Conv2d(obs_dim[0], 32, 3, stride=2)
        self.conv2 = nn.Conv2d(feat_dim, 32, 3, stride=1)
        self.conv3 = nn.Conv2d(feat_dim, 32, 3, stride=1)
        self.conv4 = nn.Conv2d(feat_dim, 32, 3, stride=1)

        # set additional layers
        self._additional_layers(dropblock)

        # apply weight init
        self.apply(weight_init)

    def _additional_layers(self, dropblock=False):
        if dropblock is True:
            # set default to 0.3
            dropblock = 0.3

        if dropblock:
            self.drop1 = DropBlock2D(block_size=7, p=dropblock)
            self.drop2 = DropBlock2D(block_size=5, p=dropblock)
            self.drop3 = DropBlock2D(block_size=3, p=dropblock)
        else:
            self.drop1 = nn.Identity()
            self.drop2 = nn.Identity()
            self.drop3 = nn.Identity()

    def forward(self, obs, scale=1.0):
        assert (obs > 1.0).sum() == 0, f"{(obs > 1.0).sum()}"

        h = self.drop1(self.act(self.conv1(obs)))
        h = self.drop2(self.act(self.conv2(h)))
        h = self.drop3(self.act(self.conv3(h)))
        h = self.act(self.conv4(h))

        norev = h.view(h.shape[0], -1)
        rev = grad_reverse(norev, scale=scale)

        return rev, norev

    def save(self, fp):
        torch.save(self.state_dict(), fp)

    def load(self, fp, cpu=False):
        map_location = torch.device("cpu") if cpu else torch.device("cuda")
        self.load_state_dict(torch.load(fp, map_location))
