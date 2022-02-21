import enum
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchaudio import datasets, transforms
import matplotlib.pyplot as plt
import itertools

# switches to graphics card if I actually had a graphics card. Might be worth an investment
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

local_lookup = (("en"), ("es"), ("fr"), ("ar"), ("zh-CN", "zh-HK", "zh-TW"))

training_data = datasets.COMMONVOICE(
    root="/Volumes/External HD/cv-corpus-8.0-2022-01-19/ar", tsv="train.tsv"
)

test_data = datasets.COMMONVOICE(
    root="/Volumes/External HD/cv-corpus-8.0-2022-01-19/ar", tsv="test.tsv"
)

train_dataloader = DataLoader(training_data)
test_dataloader = DataLoader(test_data)


class language_classification(torch.utils.data.Dataset):
    def __init__(self, training):
        if training:
            self.en = datasets.COMMONVOICE(
                root="/Volumes/External HD/cv-corpus-8.0-2022-01-19/en", tsv="train.tsv"
            )
            self.es = datasets.COMMONVOICE(
                root="/Volumes/External HD/cv-corpus-8.0-2022-01-19/es", tsv="train.tsv"
            )
            self.fr = datasets.COMMONVOICE(
                root="/Volumes/External HD/cv-corpus-8.0-2022-01-19/fr", tsv="train.tsv"
            )
            self.ar = datasets.COMMONVOICE(
                root="/Volumes/External HD/cv-corpus-8.0-2022-01-19/ar", tsv="train.tsv"
            )
            self.zh_CN = datasets.COMMONVOICE(
                root="/Volumes/External HD/cv-corpus-8.0-2022-01-19/zh-CN",
                tsv="train.tsv",
            )
            self.zh_HK = datasets.COMMONVOICE(
                root="/Volumes/External HD/cv-corpus-8.0-2022-01-19/zh-HK",
                tsv="train.tsv",
            )
            self.zh_TW = datasets.COMMONVOICE(
                root="/Volumes/External HD/cv-corpus-8.0-2022-01-19/zh-TW",
                tsv="train.tsv",
            )
        else:

            self.en = datasets.COMMONVOICE(
                root="/Volumes/External HD/cv-corpus-8.0-2022-01-19/en", tsv="test.tsv"
            )
            self.es = datasets.COMMONVOICE(
                root="/Volumes/External HD/cv-corpus-8.0-2022-01-19/es", tsv="test.tsv"
            )
            self.fr = datasets.COMMONVOICE(
                root="/Volumes/External HD/cv-corpus-8.0-2022-01-19/fr", tsv="test.tsv"
            )
            self.ar = datasets.COMMONVOICE(
                root="/Volumes/External HD/cv-corpus-8.0-2022-01-19/ar", tsv="test.tsv"
            )
            self.zh_CN = datasets.COMMONVOICE(
                root="/Volumes/External HD/cv-corpus-8.0-2022-01-19/zh-CN",
                tsv="test.tsv",
            )
            self.zh_HK = datasets.COMMONVOICE(
                root="/Volumes/External HD/cv-corpus-8.0-2022-01-19/zh-HK",
                tsv="test.tsv",
            )
            self.zh_TW = datasets.COMMONVOICE(
                root="/Volumes/External HD/cv-corpus-8.0-2022-01-19/zh-TW",
                tsv="test.tsv",
            )

        self.data = itertools.chain(
            self.en, self.es, self.fr, self.ar, self.zh_CN, self.zh_HK, self.zh_TW
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        waveform, bitrate, target_plus = self.data[index]
        return target_plus


# defines the neural network's shape
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            transforms.MelSpectrogram(),
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 5),
        )

    def forward(self, x):
        return self.layers(x)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # loops through all the batches in the dataloader
    for batch, (waveform, sample_rate, target) in enumerate(dataloader):
        # gets prediction from model and gets how correct it was
        pred = model(waveform)
        loss = loss_fn(pred, target)

        # optimizes that loss backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # prins out some neat info for progress checking
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(waveform)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for waveform, length, target in dataloader:
            pred = model(waveform)
            test_loss += loss_fn(pred, target).item()
            correct += (pred.argmax(1) == target).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss:{test_loss:>8f} \n"
        )


if __name__ == "__main__":
    # sets the neuralnetwork to use "device" model
    model = NeuralNetwork()
    print(model)

    softmax = nn.Softmax(dim=1)

    training_data = datasets.COMMONVOICE(
        root="/Volumes/External HD/cv-corpus-8.0-2022-01-19/ar", tsv="train.tsv"
    )

    test_data = datasets.COMMONVOICE(
        root="/Volumes/External HD/cv-corpus-8.0-2022-01-19/ar", tsv="test.tsv"
    )

    train_dataloader = DataLoader(training_data)
    test_dataloader = DataLoader(test_data)

    # sets a few paramaters for the training of the model
    learning_rate = 1e-3
    batch_size = 64
    epochs = 5

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n---------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")
