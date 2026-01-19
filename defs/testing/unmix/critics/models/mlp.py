import torch
import torch.nn as nn

from diffusers.models.activations import get_activation

class MLP(nn.Module):
    def __init__(self, i_dim:(int), hidden_dim:list[int], o_dim:(int), act_fn:(str)):

        self.i_dim = i_dim
        self.hidden_dim = hidden_dim
        self.o_dim = o_dim
        self.act_fn = get_activation(act_fn)

        layers = [nn.Linear(self.i_dim, self.hidden_dim[0]), self.act_fn]

        for i in range(len(self.hidden_dim) - 1):
            layers.extend([nn.Linear(self.hidden_dim[i], self.hidden_dim[i+1]), self.act_fn])

        layers.append(nn.Linear(self.hidden_dim[-1], self.o_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x:(torch.Tensor)) -> torch.Tensor:
        return self.layers(x)