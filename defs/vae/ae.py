"""
Action Item: Autoencoder (Convolutional)

Ideally: Latent dimension = 3 (for gv, npv, soil)

"""

import torch
import torch.nn as nn

# Taken from PyTorch documentation to learn about nn.Conv1d
m = nn.Conv1d(16, 33, 3, stride=1, padding=1)
input = torch.randn(20, 16, 50)
output = m(input)
print(len(input[0][0]), len(output[0][0]))