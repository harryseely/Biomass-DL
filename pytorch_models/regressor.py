import torch
from torch import nn


class Regressor(torch.nn.Module):
    def __init__(self, model, num_outputs):
        super().__init__()
        self.model = model  # model
        self.num_outputs = num_outputs  # number of classes

        #TODO: try modifyibg final output layer, perhaps with leaky relu
        self.mlp1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_outputs)
        )

        # No activation function for the final output layer following Oehmcke et al., 2021

    def forward(self, data):
        model_output = self.model(data)

        #Data enters MLP with shape Batch size x 1024
        mlp1_out = self.mlp1(model_output)

        return mlp1_out
