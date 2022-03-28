from logging import root
import torch
import slayerSNN as snn
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import learningstats
from datetime import datetime
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", dest="model")
parser.add_argument("-e", "--encoding", dest="encoding")

args = parser.parse_args()

device = torch.device("cuda")
netParams = snn.params("fashionMNIST.yaml")

logging.basicConfig(filename='fashion.log', level=logging.DEBUG)


magnitude = 10
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.5
download_value = True

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), lambda x: x*255])

transformPoisson = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), lambda x: x / magnitude])


def replicate(input, num_steps):
    ext_shape = [s for s in input.shape] + [1]
    out_shape = [s for s in input.shape] + [num_steps]
    return F.interpolate(
        input.reshape(ext_shape), size=out_shape[2:],
        mode='nearest',
    )


class NetworkInterpolationMLP(torch.nn.Module):
    def __init__(self, netParams):
        super(NetworkInterpolationMLP, self).__init__()
        self.slayer = self.slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.fc1 = self.slayer.dense((28, 28), 410)
        self.fc2 = self.slayer.dense(410, 10)

    def forward(self, input):
        x = replicate(input, netParams['simulation']['tSample'])
        x.reshape(1, 1, 28, 28, x.shape[-1])
        x = self.slayer.spike(self.slayer.psp(self.fc1(x)))
        x = self.slayer.spike(self.slayer.psp(self.fc2(x)))

        return x


class NetworkRateMLP(torch.nn.Module):
    def __init__(self, netParams):
        super(NetworkRateMLP, self).__init__()
        self.slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.fc1 = self.slayer.dense((28, 28), 410)
        self.fc2 = self.slayer.dense(410, 10)

    def forward(self, input):
        x = self.slayer.rateEncoding(input)
        x = self.slayer.spike(self.slayer.psp(self.fc1(x)))
        x = self.slayer.spike(self.slayer.psp(self.fc2(x)))

        return x


class NetworkPoissonMLP(torch.nn.Module):
    def __init__(self, netParams):
        super(NetworkPoissonMLP, self).__init__()
        self.slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.fc1 = self.slayer.dense((28, 28), 410)
        self.fc2 = self.slayer.dense(410, 10)

    def forward(self, input):
        x = self.slayer.poissonEncoding(input)
        x = self.slayer.spike(self.slayer.psp(self.fc1(x)))
        x = self.slayer.spike(self.slayer.psp(self.fc2(x)))

        return x


class NetworkInterpolationCNN(torch.nn.Module):
    def __init__(self, netParams):
        super(NetworkInterpolationCNN, self).__init__()
        self.slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.conv1 = self.slayer.conv(1, 16, 5, padding=2)
        self.pool = self.slayer.pool((2, 2))
        self.conv2 = self.slayer.conv(16, 32, 5, padding=2)

        self.fc1 = self.slayer.dense(1568, 410)
        self.fc2 = self.slayer.dense(410, 10)

    def forward(self, input):
        x = replicate(input, netParams['simulation']['tSample'])
        x = x.reshape(1, 1, 28, 28, x.shape[-1])
        x = self.slayer.spike(self.slayer.psp(self.conv1(x)))
        x = self.slayer.spike(self.slayer.psp(self.pool(x)))
        x = self.slayer.spike(self.slayer.psp(self.conv2(x)))
        x = self.slayer.spike(self.slayer.psp(self.pool(x)))
        x = self.slayer.spike(self.slayer.psp(self.fc1(x)))
        x = self.slayer.spike(self.slayer.psp(self.fc2(x)))

        return x


class NetworkRateCNN(torch.nn.Module):
    def __init__(self, netParams):
        super().__init__(NetworkRateCNN, self)
        self.slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.conv1 = self.slayer.conv(1, 16, 5, padding=2)
        self.pool = self.slayer.pool((2, 2))
        self.conv2 = self.slayer.conv(16, 32, 5, padding=2)

        self.fc1 = self.slayer.dense(1568, 410)
        self.fc2 = self.slayer.dense(410, 10)

    def forward(self, input):
        x = self.slayer.rateEncoding(input)
        x = self.slayer.spike(self.slayer.psp(self.conv1(x)))
        x = self.slayer.spike(self.slayer.psp(self.pool(x)))
        x = self.slayer.spike(self.slayer.psp(self.conv2(x)))
        x = self.slayer.spike(self.slayer.psp(self.pool(x)))
        x = self.slayer.spike(self.slayer.psp(self.fc1(x)))
        x = self.slayer.spike(self.slayer.psp(self.fc2(x)))

        return x


class NetworkPoissonCNN(torch.nn.Module):
    def __init__(self, netParams):
        super().__init__(NetworkPoissonCNN, self)
        self.slayer = snn.layer(netParams)
        self.conv1 = self.slayer.conv(1, 16, 5, padding=2)
        self.pool = self.slayer.pool((2, 2))
        self.conv2 = self.slayer.conv(16, 32, 5, padding=2)

        self.fc1 = self.slayer.dense(1568, 410)
        self.fc2 = self.slayer.dense(410, 10)

    def forward(self, input):
        x = self.slayer.poissonEncoding(input)
        x = self.slayer.spike(self.slayer.psp(self.conv1(x)))
        x = self.slayer.spike(self.slayer.psp(self.pool(x)))
        x = self.slayer.spike(self.slayer.psp(self.conv2(x)))
        x = self.slayer.spike(self.slayer.psp(self.pool(x)))
        x = self.slayer.spike(self.slayer.psp(self.fc1(x)))
        x = self.slayer.spike(self.slayer.psp(self.fc2(x)))

        return x


def main():

    model = None

    if args.encoding != "poisson":
        datasetTrain = torchvision.datasets.FashionMNIST(
            root=netParams['training']['path']['train'], transform=transform, download=download_value)
        datasetTest = torchvision.datasets.FashionMNIST(
            root=netParams['training']['path']['test'], transform=transform, train=False, download=download_value)
    else:
        datasetTrain = torchvision.datasets.FashionMNIST(
            root=netParams['training']['path']['train'], transform=transformPoisson, download=download_value)
        datasetTest = torchvision.datasets.FashionMNIST(
            root=netParams['training']['path']['test'], transform=transform, train=False, download=download_value)

    if args.encoding == "poisson":
        if args.model == "CNN":
            model = NetworkPoissonCNN(netParams)
        else:
            model = NetworkPoissonMLP(netParams)

    if args.encoding == "rate":
        if args.model == "CNN":
            model = NetworkRateCNN(netParams)
        else:
            model = NetworkRateMLP(netParams)

    if args.encoding == "interpolation":
        if args.model == "CNN":
            model = NetworkInterpolationCNN(netParams)
        else:
            model = NetworkInterpolationMLP(netParams)

    if model == None:
        raise Exception(
            "The parsing of the arguments failed. Make sure to set the encoding flag")

    # Load datasets
    loadedTrain = DataLoader(datasetTrain, batch_size=1,
                             num_workers=0, shuffle=False)
    loadedTest = DataLoader(datasetTest, batch_size=1,
                            num_workers=0, shuffle=False)

    # Send the model to the GPU
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    criterion = snn.loss(netParams).to(device)

    stats = learningstats.learningStats()

    for epoch in range(NUM_EPOCHS):

        tSt = datetime.now()

        # Trainign loop
        for i, (sample, label) in enumerate(loadedTrain):

            sample = sample.to(device)

            # Forward pass
            output = model(sample)

            # Create desired spike train
            desired = torch.zeros((10, 1, 1, 1))
            desired[label, ...] = 1

            # Get the loss
            loss = criterion.numSpikes(output, desired)

            # Zero the gradients for the optimizer
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Compute statistics
            stats.training.numSamples += len(sample)
            stats.training.correctSamples += torch.sum(
                snn.predict.getClass(output) == label).data.item()
            stats.training.lossSum += loss.cpu().data.item()

            if i % 100:
                stats.print(epoch, i, (datetime.now() - tSt).total_seconds())

        # Save the network
        torch.save(model.state_dict(), "netowork" + str(epoch))
        torch.save(model.state_dict(), "optimizer" + str(epoch))

        # Testing loop
        for i, (sample, label) in enumerate(loadedTest):
            sample = sample.to(device)

            desired = torch.zeros((10, 1, 1, 1))
            desired[label, ...] = 1

            output = model(sample)

            loss = criterion.numSpikes(output, desired)

            stats.testing.correctSamples += torch.sum(
                snn.predict.getClass(output) == label).data.item()
            stats.testing.numSamples += len(label)

            if i % 100:
                stats.print(epoch, i, (datetime.now() - tSt).total_seconds())


if __name__ == "__main__":
    main()
