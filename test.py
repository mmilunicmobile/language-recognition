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
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        pass


training_data = datasets.COMMONVOICE(
    root="/Volumes/External HD/cv-corpus-8.0-2022-01-19/zh-TW", tsv="train.tsv"
)

print(next(iter(training_data))[2]["sentence"])
