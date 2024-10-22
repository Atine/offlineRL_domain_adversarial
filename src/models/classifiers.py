import torch
import torch.nn as nn


from src.models import weight_init


class ClassifierWithLogitsRev(nn.Module):
    def __init__(self, repr_dim, class_num, feature_dim, hidden_dim):
        super().__init__()
        self.class_num = class_num

        self.fc1 = nn.Linear(repr_dim, feature_dim)
        self.fc2 = nn.Linear(feature_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, class_num)

        # authors default init (from encoder)
        self.apply(weight_init)

    def forward(self, obs):
        x = nn.functional.relu(self.fc1(obs))
        x = nn.functional.relu(self.fc2(x))
        logits = self.fc3(x)

        if self.class_num == 1:
            x = torch.sigmoid(logits)
        else:
            x = torch.nn.functional.log_softmax(logits, dim=1)

        return x, logits

    def save(self, fp):
        torch.save(self.state_dict(), fp)

    def load(self, fp, cpu=False):
        map_location = torch.device("cpu") if cpu else torch.device("cuda")
        self.load_state_dict(torch.load(fp, map_location))
