import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchaudio import datasets, transforms
import torchaudio
import torchvision
import matplotlib.pyplot as plt
import itertools
import numpy as np
import torch.nn.functional as F


# creates a custom mod of the itertools.chain class. Has subscriptability and length defined
class chain_subscript:
    def __init__(self, *iterables):
        self.iterables = iterables
        self.iterables_starts = [0]
        for i in range(1, len(iterables) + 1):
            self.iterables_starts.append(
                self.iterables_starts[i - 1] + len(iterables[i - 1])
            )
        self.length = self.iterables_starts[-1]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        for i in range(len(self.iterables)):
            if self.iterables_starts[i + 1] > index >= self.iterables_starts[i]:
                return self.iterables[i][index - self.iterables_starts[i]]
        raise IndexError("list index out of range")


# defines the dataset I will be using
class language_classification(torch.utils.data.Dataset):
    def __init__(self, testing=False):
        self.local_lookup = {
            "en": 0,
            "es": 1,
            "fr": 2,
            "ar": 3,
            "zh-CN": 4,
            "zh-HK": 4,
            "zh-TW": 4,
        }

        trast = "test.tsv" if testing else "train.tsv"  # to train or to test

        self.en = datasets.COMMONVOICE(
            root="/Volumes/External HD/cv-corpus-8.0-2022-01-19/en", tsv=trast
        )
        self.es = datasets.COMMONVOICE(
            root="/Volumes/External HD/cv-corpus-8.0-2022-01-19/es", tsv=trast
        )
        self.fr = datasets.COMMONVOICE(
            root="/Volumes/External HD/cv-corpus-8.0-2022-01-19/fr", tsv=trast
        )
        self.ar = datasets.COMMONVOICE(
            root="/Volumes/External HD/cv-corpus-8.0-2022-01-19/ar", tsv=trast
        )
        self.zh_CN = datasets.COMMONVOICE(
            root="/Volumes/External HD/cv-corpus-8.0-2022-01-19/zh-CN",
            tsv=trast,
        )
        self.zh_HK = datasets.COMMONVOICE(
            root="/Volumes/External HD/cv-corpus-8.0-2022-01-19/zh-HK",
            tsv=trast,
        )
        self.zh_TW = datasets.COMMONVOICE(
            root="/Volumes/External HD/cv-corpus-8.0-2022-01-19/zh-TW",
            tsv=trast,
        )

        self.data = chain_subscript(
            self.en, self.es, self.fr, self.ar, self.zh_CN, self.zh_HK, self.zh_TW
        )

    # allows us to get the length of the data
    def __len__(self):
        return len(self.data)

    # allows us to get the item at the index of the data (allows us to do most things)
    def __getitem__(self, index):
        waveform, sample_rate, target_plus = self.data[index]

        target = self.local_lookup[target_plus["locale"]]

        sound_data = torchaudio.transforms.Resample(sample_rate, 8000)(waveform)[0]

        temp_data = torch.zeros([40000])

        # formats the length of the data
        if len(sound_data) < 40000:
            # adds zeros if it is too short
            temp_data[: len(sound_data)] = sound_data[:]
        else:
            temp_data[:] = sound_data[:40000]

        sound_data = temp_data

        sound_formatted = torch.zeros([10000])
        sound_formatted[:10000] = sound_data[::4]
        sound_formatted = sound_formatted.reshape(1, 10000)
        mel_spectrogram = transforms.MelSpectrogram(8000)(sound_formatted)
        return mel_spectrogram, target


# defines the neural network's shape
class neural_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6528, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    for batch, data in enumerate(dataloader):
        # gets prediction from model and gets how correct it was
        inputs, targets = data

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, targets)

        # optimizes that loss backwards
        loss.backward()
        optimizer.step()

        # prins out some neat info for progress checking
        if batch % 1000 == 0:
            current_loss, current = loss.item(), batch * len(data)
            print(f"loss: {current_loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataset, model, loss_fn):
    size = len(dataset)
    num_batches = size // batch_size
    test_loss, correct = 0, 0

    with torch.no_grad():
        for spectrogram, target in dataset:
            pred = model(spectrogram)
            test_loss += loss_fn(pred, target).item()
            correct += (pred.argmax(1) == target).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss:{test_loss:>8f} \n"
        )


def subsample_random(dataset, length):
    return Subset(dataset, torch.randperm(len(dataset))[:length])


if __name__ == "__main__":
    # switches to graphics card if I actually had a graphics card. Might be worth an investment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    softmax = nn.Softmax(dim=1)

    # sets a few paramaters for the training of the model
    learning_rate = 1e-3
    batch_size = 1
    epochs = 5

    training_data = subsample_random(language_classification(), 50000)
    testing_data = subsample_random(language_classification(True), 10000)

    training_data_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
    )
    testing_data_loader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

    # sets the neuralnetwork to use "device" model
    model = neural_network()
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Starting epoch {t}")
        train_loop(training_data_loader, model, loss_fn, optimizer)
        test_loop(testing_data_loader, model, loss_fn)

    print("Done learning!")

    torch.save(model.state_dict(), "model.pth")

    print("Saved")
