# Preamble

import torch

# NN classes

# MLP to infer GV, NPV and soil fractions from a spectrum
class FractionMLP:
    """
    Will have 2 layers to start off with.
        - W1: weights for layer 1
        - b1: bias for layer 1
        - W2: weights for layer 2
        - b2: bias for layer 2
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialisation, we will have inputs ranging from 400-2490 nm as requested:
            - want input_size = 209
            - want output_size = 3 for gv, npv, soil
            - hidden_size can be free for now

        1 hidden layer for now
        """
        self.W1 = torch.randn(input_size, hidden_size, requires_grad=True)
        self.b1 = torch.randn(1, hidden_size, requires_grad=True)
        self.W2 = torch.randn(hidden_size, output_size, requires_grad=True)
        self.b2 = torch.randn(1, output_size, requires_grad=True)

    def forward(self, X):
        """
        Forward pass. Will use sigmoid function for now.
        """
        self.z1 = torch.matmul(X, self.W1) + self.b1
        self.a1 = torch.sigmoid(self.z1)  # Hidden layer activation
        self.z2 = torch.matmul(self.a1, self.W2) + self.b2
        self.a2 = torch.sigmoid(self.z2)  # Output layer activation
        return self.a2