import torch
import torch.nn as nn

from diffusers.models.activations import get_activation

class MLP(nn.Module):
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    def __init__(self, hidden_dim:list[int], i_dim:(int)=210, o_dim:(int)=3, dropout:(float)=0.1, leaky_slope:(float)=0.2):
=======
    def __init__(self, hidden_dim:list[int], i_dim:(int)=81, o_dim:(int)=3, dropout:(float)=0.1, leaky_slope:(float)=0.2):
>>>>>>> b04a5651f926d8053e527722711cc244d89e3902
=======
    def __init__(self, hidden_dim:list[int], i_dim:(int)=81, o_dim:(int)=3, dropout:(float)=0.1, leaky_slope:(float)=0.2):
>>>>>>> b04a5651f926d8053e527722711cc244d89e3902
=======
    def __init__(self, hidden_dim:list[int], i_dim:(int)=81, o_dim:(int)=3, dropout:(float)=0.1, leaky_slope:(float)=0.2):
>>>>>>> b04a5651f926d8053e527722711cc244d89e3902
=======
    def __init__(self, hidden_dim:list[int], i_dim:(int)=81, o_dim:(int)=3, dropout:(float)=0.1, leaky_slope:(float)=0.2):
>>>>>>> b04a5651f926d8053e527722711cc244d89e3902
        super().__init__()

        self.i_dim = i_dim
        self.hidden_dim = hidden_dim
        self.o_dim = o_dim
        self.dropout = nn.Dropout(p=dropout) 
        self.leaky = nn.LeakyReLU(negative_slope=leaky_slope)

        # Initializes layer list, with the first one
        layers = [nn.Linear(self.i_dim, self.hidden_dim[0]), nn.BatchNorm1d(self.hidden_dim[0]), self.leaky, self.dropout]
        # Hidden layers
        for i in range(len(self.hidden_dim) - 1):
            layers.extend([nn.Linear(self.hidden_dim[i], self.hidden_dim[i+1]), nn.BatchNorm1d(self.hidden_dim[i+1]), self.leaky, self.dropout])
        # Final linear layer to output dimension
        layers.append(nn.Linear(self.hidden_dim[-1], self.o_dim))
        # Compiles list into sequential module
        self.layers = nn.Sequential(*layers)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x:(torch.Tensor)) -> torch.Tensor:
        x = self.layers(x) # All layers applied
        x = torch.relu(x) # Applies relu, destroys negative vals
        return x / (torch.sum(x, dim=1, keepdim=True) + 1e-12) # Normalize to sum to 1 along dim 1 (ab dim)