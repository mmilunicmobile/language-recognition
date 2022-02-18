import torch
from torch import nn
from torch.utils.data import DataLoader
from torchaudio import datasets, transforms
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


training_data = datasets.COMMONVOICE(root="ml", tsv="train.tsv")

print(next(iter(training_data))[2])
