import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "Augment1",
    "augmentator",
]


class Augment1(object):
    """wrapper for differentl augmentations: Augment1

    RandomShiftsAug()

    """

    def __init__(self, frame_stack):
        self.aug1 = RandomShiftsAug(pad=4)

    def __call__(self, x, shift):
        # aug1
        x, shifted = self.aug1(x, shift)

        # return
        return x, shifted


class RandomShiftsAug(nn.Module):
    def __init__(self, pad=4):
        super().__init__()
        self.pad = pad

    def forward(self, x, shift=None):
        n, c, h, w = x.size()
        assert h == w

        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)

        arange = torch.linspace(
            -1.0 + eps,
            1.0 - eps,
            h + 2 * self.pad,
            device=x.device,
            dtype=x.dtype,
        )[:h]

        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        if shift is None:
            shift = torch.randint(
                0,
                2 * self.pad + 1,
                size=(n, 1, 1, 2),
                device=x.device,
                dtype=x.dtype,
            )
            shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return (
            F.grid_sample(x, grid, padding_mode="zeros", align_corners=False),
            shift,
        )


augmentator = {
    1: Augment1,
}
