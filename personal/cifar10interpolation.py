import slayerSNN as snn
import numpy as np
from slayerSNN import slayer
import torch
from torchvision.datasets import CIFAR10
import torchvision
from torch.utils.data import DataLoader
from encoding import Encoding
from learningstats import learningStats
import torchvision.models as models
from datetime import date, datetime
from torch.profiler import profile, record_function, ProfilerActivity
import logging
logging.basicConfig(filename='rccifar10.log',
                    encoding='utf-8', level=logging.DEBUG)


netParams = snn.params("network.yaml")
device = torch.device("cuda")

num_epochs = 10
pixel_to_time_index = {}
load = False

transformation = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), lambda x: x*255])


def calculate_spike_times():
    encode = Encoding(netParams['simulation']['tSample'])
    for i in range(256):
        pixel_to_time_index[i] = encode._calculate_spike_times(i)


def image_to_spike_tensor(input: torch.Tensor, empty_array: torch.Tensor, Ts: int):

    # # empty_array[cEvent, xEvent, yEvent, tEvent] = 1/Ts
    for B in range(input.shape[0]):
        for C in range(input.shape[1]):
            for H in range(input.shape[2]):
                for W in range(input.shape[3]):
                    pixel = np.array(
                        pixel_to_time_index[int(input[B][C][H][W])])
                    # empty_array[C,H,W, pixel] = 1/Ts
                    empty_array[B][C][H][W][pixel] = 1/Ts

    return empty_array.to(device)


class Network(torch.nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.fc1 = self.slayer.dense((32, 32, 3), 240)
        self.fc2 = self.slayer.dense(240, 10)
        # self.fc3 = self.slayer.dense(80, 10)
        self.nTimeBins = int(
            netParams['simulation']['tSample'] / netParams['simulation']['Ts'])
        self.timeStep = int(netParams['simulation']['Ts'])

        self.pspLayer = self.slayer.pspLayer()

    def forward(self, input):
        spikes = image_to_spike_tensor(input, torch.zeros(
            (1, 3, 32, 32, self.nTimeBins)), self.timeStep)
        if int(torch.sum(spikes)) != int(torch.sum(input)):
            raise Exception("Error in conversion")

        # print(torch.sum(spikes))
        # print(spikes.shape)
        # spikes = self.slayer.spike(self.pspLayer(spikes))
        # print(torch.sum(spikes))
        layer1 = self.slayer.spike(self.slayer.psp(self.fc1(spikes)))
        layer2 = self.slayer.spike(self.slayer.psp(self.fc2(layer1)))
        # layer3 = self.slayer.spike(self.slayer.psp(self.fc3(layer2)))
        # layer1 = self.slayer.spike(self.fc1(self.slayer.psp(spikes)))
        # layer2 = self.slayer.spike(self.fc2(self.slayer.psp(layer1)))

        return layer2


def overfit_single_batch():
    dataset_train = CIFAR10(root="", download=False,
                            transform=transformation, train=True)
    dataset_test = CIFAR10(root="", download=False,
                           transform=transformation, train=False)

    loaded_train = DataLoader(
        dataset_train, batch_size=1, num_workers=0, shuffle=False)
    loaded_test = DataLoader(dataset_test, batch_size=1,
                             num_workers=0, shuffle=False)

    print("Finish loading data")

    print("Computing pixel time indexes")
    calculate_spike_times()
    print("Finish calculating pixel spike times")

    network = Network().to(device)
    criterion = snn.loss(netParams).to(device)

    if load == True:
        network.load_state_dict(torch.load("network1"))

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001, amsgrad=True)

    if load == True:
        optimizer.load_state_dict(torch.load("optimizer1"))

    stats = learningStats()

    (sample, label) = next(iter(loaded_train))
    sample = sample.to(device)
    label = label.to(device)

    desired = torch.zeros((10, 1, 1, 1))
    desired[label, ...] = 1

    for i in range(50):
        print("______________________")
        print("EPOCH: " + str(i))
        output = network.forward(sample)
        print("For the first layer of the SNN")
        print(torch.sum(network.fc1.weight > 0))
        print(torch.sum(network.fc1.weight))

        print("For the second layer of the SNN")
        print(torch.sum(network.fc2.weight))
        print("For the third layer of the SNN")
        print(torch.sum(network.fc3.weight))
        loss = criterion.numSpikes(output, desired)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'The loss is {loss.item()}')
        print("______________________")

        # print("The number of times the correct neuron fired: " + str(int(torch.sum(output[0][label][0][0]))))
        for i in range(10):
            print(
                f"The number of times neuron  {i} fired: " + str(int(torch.sum(output[0][i][0][0]))))

        print("______________________")



if __name__ == "__main__":
    dataset_train = CIFAR10(root="", download=False,
                            transform=transformation, train=True)
    dataset_test = CIFAR10(root="", download=False,
                           transform=transformation, train=False)

    loaded_train = DataLoader(
        dataset_train, batch_size=1, num_workers=0, shuffle=False)
    loaded_test = DataLoader(dataset_test, batch_size=1,
                             num_workers=0, shuffle=False)

    logging.info("Finish loading data")

    logging.info("Computing pixel time indexes")
    calculate_spike_times()
    logging.info("Finish calculating pixel spike times")

    network = Network().to(device)
    criterion = snn.loss(netParams).to(device)

    if load == True:
        network.load_state_dict(torch.load("network1"))

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001, amsgrad=True)

    if load == True:
        optimizer.load_state_dict(torch.load("optimizer1"))

    stats = learningStats()

    for epoch in range(num_epochs):
        time_start = datetime.now()

        for i, (sample, label) in enumerate(loaded_train):
            sample.to(device)
            label.to(device)
            desired = torch.zeros((10, 1, 1, 1))
            desired[int(label), ...] = 1
            desired = desired.to(device)

            output = network(sample)

            loss = criterion.numSpikes(output, desired)

            stats.training.correctSamples += torch.sum(
                snn.predict.getClass(output) == label).data.item()
            stats.training.numSamples += len(label)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if i % 50:
                logging.debug("The sum of weight in layer 1" +
                              str(int(torch.sum(network.fc1.weight))))
                logging.debug("The sum of weight in layer 2" +
                              str(int(torch.sum(network.fc2.weight))))
                for j in range(10):
                    logging.debug(
                        f"The number of times neuron {j} fired" + str(int(torch.sum(output[0][j][0][0]))))
                logging.debug("The label for the class is: " + str(int(label)))

            stats.training.lossSum += loss.cpu().data.item()

            if i % 100 == 0:
                stats.print(epoch, i, (datetime.now() -
                            time_start).total_seconds())

        torch.save(network.state_dict(), "network" + epoch)
        torch.save(optimizer.state_dict(), "optimizer" + epoch)
        logging.info("Starting the testing")

        for i, (input, label) in enumerate(loaded_test, 0):
            input = input.to(device)
            # target = label.to(device)

            output = network.forward(input)

            stats.testing.correctSamples += torch.sum(
                snn.predict.getClass(output) == label).data.item()
            stats.testing.numSamples += len(label)

            loss = criterion.numSpikes(output, label)
            stats.testing.lossSum += loss.cpu().data.item()
            if i % 100 == 0:
                stats.print(epoch, i)
