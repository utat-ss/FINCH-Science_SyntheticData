import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, h_dim=[128, 128], out_dim=3):
        super().__init__()
        self.spec_dim = in_dim
        self.denoiser_hidden_dim = h_dim
        self.out_dim = out_dim

        denoiser_layers = []

        denoiser_layers.extend([
            nn.Linear(self.spec_dim, self.denoiser_hidden_dim[0]),
            nn.ReLU()
        ])

        for i in range(0, len(self.denoiser_hidden_dim) - 1): # Appending hidden layers of denoiser
            denoiser_layers.extend([
                nn.Linear(self.denoiser_hidden_dim[i], self.denoiser_hidden_dim[i+1]),
                nn.ReLU()
            ])

        denoiser_layers.extend([ 
            nn.Linear(self.denoiser_hidden_dim[-1], self.out_dim),
            # nn.Sigmoid()
        ]) # Final output layer

        self.net = nn.Sequential(*denoiser_layers)

    def forward(self, x):
        return self.net(x)

