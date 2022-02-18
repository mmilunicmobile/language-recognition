import torch
from torch import nn
from torch.utils.data import DataLoader
from torchaudio import datasets, transforms
import matplotlib.pyplot as plt

# switches to graphics card if I actually had a graphics card. Might be worth an investment
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# defines the neural network's shape
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 5),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# sets the neuralnetwork to use "device" model
model = NeuralNetwork().to(device)
print(model)

softmax = nn.Softmax(dim=1)

training_data = datasets.COMMONVOICE(
    root="/Volumes/External HD/cv-corpus-8.0-2022-01-19/ar", tsv="train.tsv"
)

# sets a few paramaters for the training of the model
learning_rate = 1e-3
batch_size = 64
epochs = 5

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print(len(training_data))
