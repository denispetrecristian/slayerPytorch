from turtle import forward
import torch
import slayerSNN as snn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
import learningstats
import torchvision
import torch.nn.functional as F
from datetime import datetime

transformation = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), lambda x: x*255])

netParams = snn.params("interpolateNetwork.yaml")
device = torch.device("cuda")


def replicate(input, num_steps):
    ext_shape = [s for s in input.shape] + [1]
    out_shape = [s for s in input.shape] + [num_steps]
    return F.interpolate(
        input.reshape(ext_shape), size=out_shape[2:],
        mode='nearest',
    )


class Network2(torch.nn.Module):
    def __init__(self, netParams):
        super(Network, self).__init__()
        self.slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        # self.fc1 = self.slayer.dense((32*32 * 3), 512)
        # self.fc2 = self.slayer.dense(512, 10)
        self.conv1 = self.slayer.conv(3,6,5)
        self.pool1 = self.slayer.pool((2,2))
        self.conv2 = self.slayer.conv(6,16,5)
        self.fc1 = self.slayer.dense((16*10*10), 300)
        self.fc2 = self.slayer.dense(300, 150)
        self.fc3 = self.slayer.dense(150, 10)

        self.nTimeBins = int(
            netParams['simulation']['tSample'] / netParams['simulation']['Ts'])
        self.timeStep = int(netParams['simulation']['Ts'])

    def forward(self, input):
        x = replicate(input, self.nTimeBins)
        x = x.reshape(1, 3, 32, 32, x.shape[-1])
        # x = self.slayer.spike(self.slayer.psp(self.fc1(x)))
        # x = self.slayer.spike(self.slayer.psp(self.fc2(x)))
        x = self.slayer.spike(self.slayer.psp(self.conv1(x)))
        x = self.slayer.spike(self.slayer.psp(self.pool1(x)))
        x = self.slayer.spike(self.slayer.psp(self.conv2(x)))
        x = x.reshape(1,-1,1,1,x.shape[-1])
        x = self.slayer.spike(self.slayer.psp(self.fc1(x)))
        x = self.slayer.spike(self.slayer.psp(self.fc2(x)))
        x = self.slayer.spike(self.slayer.psp(self.fc3(x)))

        return x

class Network(torch.nn.Module):
    def __init__(self, netParams):
        super(Network, self).__init__()
        self.slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.fc1 = self.slayer.dense((32*32 * 3), 410)
        self.fc2 = self.slayer.dense(410, 10)
        # self.fc3 = self.slayer.dense(256,10)

        self.nTimeBins = int(
            netParams['simulation']['tSample'] / netParams['simulation']['Ts'])
        self.timeStep = int(netParams['simulation']['Ts'])

    def forward(self, input):
        x = replicate(input, self.nTimeBins)
        x = x.reshape(1, -1, 1, 1, x.shape[-1])
        x = self.slayer.spike(self.slayer.psp(self.fc1(x)))
        x = self.slayer.spike(self.slayer.psp(self.fc2(x)))
        # x = self.slayer.spike(self.slayer.psp(self.fc3(x)))

        return x


def main():
    dataset_train = CIFAR10(root="", download=False,
                            transform=transformation, train=True)
    dataset_test = CIFAR10(root="", download=False,
                           transform=transformation, train=False)

    loaded_train = DataLoader(
        dataset_train, batch_size=1, num_workers=0, shuffle=False)
    loaded_test = DataLoader(dataset_test, batch_size=1,
                             num_workers=0, shuffle=False)

    network = Network(netParams).to(device)
    optimizer = torch.optim.Adam(
        network.parameters(), amsgrad=True, lr=1e-3, weight_decay=0.7)

    criterion = snn.loss(netParams).to(device)
    stats = learningstats.learningStats()

    for epoch in range(20):

        tSt = datetime.now()

        for i, (sample, label) in enumerate(loaded_train):
            sample = sample.to(device)
            desired = torch.zeros((10, 1, 1, 1)).to(device)
            desired[label, ...] = 1

            output = network(sample)
            loss = criterion.numSpikes(output, desired)

            stats.training.correctSamples += torch.sum(
                snn.predict.getClass(output) == label).data.item()
            stats.training.numSamples += len(label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            stats.training.lossSum += loss.cpu().data.item()

            if i % 10 == 0:
                stats.print(epoch, i, (datetime.now() - tSt).total_seconds())

        for i, (sample, label) in enumerate(loaded_test):
            sample = sample.to(device)
            desired = torch.zeros((10, 1, 1, 1))
            desired[label, ...] = 1

            output = network(sample)
            loss = criterion.numSpikes(output, desired)

            stats.testing.correctSamples += torch.sum(
                snn.predict.getClass(output) == label).data.item()
            stats.testing.numSamples += len(label)

            stats.testing.lossSum += loss.cpu().data.item()

            if i % 10 == 0:
                stats.print(epoch, i, (datetime.now() - tSt).total_seconds())

        stats.update()
        fileName = "interpolationCifar10" + str(epoch)
        stats.save(filename= fileName)


if __name__ == "__main__":
    main()
