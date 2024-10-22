import torch.nn as nn


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        # nn.init.normal_(m.weight.data, mean=0.0, std=0.1)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        # nn.init.normal_(m.weight.data, mean=0.0, std=0.1)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
