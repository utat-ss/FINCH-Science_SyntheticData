"""
Guide used to assist in creating this MLP
Guide: https://medium.com/@mn05052002/building-a-simple-mlp-from-scratch-using-pytorch-7d50ca66512b 
"""

# Preamble

import torch
import torch.nn as nn

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
        # Initially set up as float32, changed to float64
        self.W1 = torch.randn(input_size, hidden_size, dtype=torch.float64, requires_grad=True)
        self.b1 = torch.randn(1, hidden_size, requires_grad=True).to(torch.float64)
        self.W2 = torch.randn(hidden_size, output_size, requires_grad=True).to(torch.float64)
        self.b2 = torch.randn(1, output_size, requires_grad=True).to(torch.float64)

    def forward(self, X):
        """
        Forward pass. Will use sigmoid function for now.
        """
        self.z1 = torch.matmul(X, self.W1) + self.b1
        self.a1 = torch.sigmoid(self.z1)  # Hidden layer activation
        self.z2 = torch.matmul(self.a1, self.W2) + self.b2
        self.a2 = torch.sigmoid(self.z2)  # Output layer activation
        # self.a2 = self.z2
        return self.a2
    
    def backward(self, X, y, output, lr=0.01):
        """
        Backpropagation function. 
            - Needs to take X as a torch tensor object.
        """
        m = X.shape[0]
        dz2 = output - y # loss
        dW2 = torch.matmul(self.a1.T, dz2)
        db2 = torch.sum(dz2, axis=0) / m

        da1 = torch.matmul(dz2, self.W2.T)
        dz1 = da1*(self.a1*(1-self.a1))
        dw1 = torch.matmul(X.T, dz1) / m
        db1 = torch.sum(dz1, axis=0) / m
        
        with torch.no_grad():
            self.W1 -= lr * dw1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

    def train(self, X, y, epochs=1000, lr=0.01):
        """
        Training function. Returns the losses to graph if needed.
        """
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            #Compute loss using (Mean Squared Error)
            loss = torch.mean((output - y) ** 2)
            losses.append(loss.item())
            #update weights
            self.backward(X, y, output, lr)
        return losses
    

# Adopted from Andrew's model.py file

class MLP(nn.Module):
    """
    Trials with modifications to Andrew's original model.
    """


    def __init__(self, in_dim, h_dim=[128, 128], out_dim=3):
        super().__init__()
        self.spec_dim = in_dim
        self.denoiser_hidden_dim = h_dim
        self.out_dim = out_dim

        denoiser_layers = []

        denoiser_layers.extend([
            nn.Linear(self.spec_dim, self.denoiser_hidden_dim[0]),
            nn.SiLU()
        ]) # Init layer of the denoiser, takes in embeddings

        for i in range(0, len(self.denoiser_hidden_dim) - 1): # Appending hidden layers of denoiser
            denoiser_layers.extend([
                nn.Linear(self.denoiser_hidden_dim[i], self.denoiser_hidden_dim[i+1]),
                nn.SiLU()
            ])

        denoiser_layers.extend([
            nn.Linear(self.denoiser_hidden_dim[-1], self.out_dim),
            nn.SiLU()
        ])

        # denoiser_layers.append(nn.Linear(self.denoiser_hidden_dim, self.denoiser_hidden_dim))

        self.net = nn.Sequential(*denoiser_layers)
        
        # self.net = nn.Sequential(
        #     nn.Linear(in_dim, h_dim),
        #     nn.ReLU(),
        #     nn.Linear(h_dim, h_dim),
        #     nn.ReLU(),
        #     nn.Linear(h_dim, h_dim),
        #     nn.Sigmoid(),
        #     nn.Linear(h_dim, out_dim)
        # )

    def forward(self, x):
        return self.net(x)


