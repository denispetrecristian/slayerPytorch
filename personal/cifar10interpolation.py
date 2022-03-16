from cmath import sqrt
from attr import validate
import slayerSNN as snn
import numpy as np
from slayerSNN import slayer
import torch
from torchvision.datasets import CIFAR10
import torchvision
from torch.utils.data import DataLoader, random_split
from encoding import Encoding
from learningstats import learningStats
import torchvision.models as models
from datetime import date, datetime
from torch.profiler import profile, record_function, ProfilerActivity
import logging


logging.basicConfig(filename='rccifar10.log', level=logging.DEBUG)


netParams = snn.params("network.yaml")
device = torch.device("cuda")

num_epochs = 10
pixel_to_time_index = {}
load = False

BATCH_SIZE = 8

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


def log_sample_properties(sample: torch.Tensor):
    for B in range(sample.shape[0]):
        i = 0
        for C in range(sample.shape[1]):
            for H in range(sample.shape[2]):
                for W in range(sample.shape[3]):
                    logging.debug(
                        f"For the {i} neuron there are {torch.sum(sample[B][C][H][W])} spikes")
                    i += 1


class Network(torch.nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.fc1 = self.slayer.dense((32, 32, 3), 410, weightScale=1)
        self.fc2 = self.slayer.dense(410, 10, weightScale=1)

        # Initialize layers
        # torch.nn.init.uniform(self.fc1.weight, 1/300, 1/90)
        # torch.nn.init.uniform(self.fc2.weight, 1/40, 1/120)

        self.nTimeBins = int(
            netParams['simulation']['tSample'] / netParams['simulation']['Ts'])
        self.timeStep = int(netParams['simulation']['Ts'])

        self.pspLayer = self.slayer.pspLayer()

    def forward(self, input):
        spikes = image_to_spike_tensor(input, torch.zeros(
            (1, 3, 32, 32, self.nTimeBins)), self.timeStep)
        if int(torch.sum(spikes)) != int(torch.sum(input)):
            raise Exception("Error in conversion")

        layer1 = self.slayer.spike(self.slayer.psp(self.fc1(spikes)))
        layer2 = self.slayer.spike(self.slayer.psp(self.fc2(layer1)))

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

    optimizer = torch.optim.Adam(
        network.parameters(), amsgrad=True, lr=1e-4, weight_decay=1e-4)
    # optimizer = torch.optim.Adagrad(network.parameters(), lr=0.01)

    if load == True:
        optimizer.load_state_dict(torch.load("optimizer1"))

    stats = learningStats()

    (sample, label) = next(iter(loaded_train))
    sample = sample.to(device)
    label = label.to(device)

    desired = torch.zeros((10, 1, 1, 1)).to(device)
    desired[label, ...] = 1

    for i in range(5000):

        output = network.forward(sample)
        loss = criterion.numSpikes(output, desired)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'The loss is {loss.item()}')
        print("______________________")

        # print("The number of times the correct neuron fired: " + str(int(torch.sum(output[0][label][0][0]))))
        fire = torch.sum(output[..., 0:2550], 4, keepdim=True)
        for j in range(10):
            print(f"The neuron {j} fired " + str(int(fire[0][j][0][0])))

        print("______________________")


# overfit_single_batch()

def validate_hyperparameters():
    dataset_cifar10 = CIFAR10(root="", download=False,
                              transform=transformation, train=True)
    dataset_train, dataset_validation, dump_dataset = random_split(
        dataset_cifar10, [10000, 10000, 30000])

    loaded_train = DataLoader(
        dataset_train, batch_size=1, num_workers=0, shuffle=False)
    loaded_validation = DataLoader(
        dataset_validation, batch_size=1, num_workers=0, shuffle=False)

    calculate_spike_times()

    network = Network().to(device)
    criterion = snn.loss(netParams).to(device)
    optimizer = torch.optim.Adam(
        network.parameters(), lr=1e-3, weight_decay=0.3, amsgrad=True)

    stats = learningStats()

    logging_time = 20

    for epoch in range(num_epochs):
        tSt = datetime.now()
        network.train()

        for i, (sample, label) in enumerate(loaded_train):
            sample = sample.to(device)

            desired = torch.zeros((10, 1, 1, 1)).to(device)
            desired[label, ...] = 1

            output = network(sample).to(device)

            stats.training.correctSamples += torch.sum(
                snn.predict.getClass(output) == label).data.item()
            stats.training.numSamples += len(label)

            loss = criterion.numSpikes(output, desired)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            stats.training.lossSum += loss.cpu().data.item()

            if i % logging_time == 0:
                stats.print(epoch, i, (datetime.now() - tSt).total_seconds())
                print("Current loss is : " + str(loss.item()))

            # if i % 50 == 0:
            #     log_sample_properties(sample)

            if i % logging_time == 0:
                fire = torch.sum(output[..., :], 4, keepdim=True)
                logging.debug(f"The label of the sample is: {int(label)}")
                for j in range(10):
                    logging.debug(
                        f"The neuron {j} fired " + str(int(fire[0][j][0][0])))
                logging.debug("The average of the weights in the first layer")
                logging.debug(
                    float(torch.sum(network.fc1.weight) / torch.numel(network.fc2.weight)))
                logging.debug("The average of the weights in the second layer")
                logging.debug(
                    float(torch.sum(network.fc2.weight) / torch.numel(network.fc1.weight)))

            if i % logging_time == 0:
                spikes = image_to_spike_tensor(sample, torch.zeros(
                    (1, 3, 32, 32, network.nTimeBins)), 1)
                res = network.slayer.spike(
                    network.slayer.psp(network.fc1(spikes)))
                res2 = network.slayer.psp(network.fc2(res))

                for i in range(10):
                    avg = float(torch.sum(res2[0][i][0][0])) / \
                        float(torch.numel(res2[0][i][0][0]))
                    logging.debug(
                        f"The average membrane potential for neuron {i} is {avg}")

        network.eval()

        with torch.no_grad():
            for i, (sample, label) in enumerate(loaded_validation):
                sample = sample.to(device)
                label = label.to(device)
                desired = torch.zeros((10, 1, 1, 1))
                desired[label, ...] = 1

                output = network(sample)

                loss = criterion.numSpikes(output, desired)

                stats.testing.lossSum += loss.cpu().data.item()
                if i % 10 == 0:
                    stats.print(epoch, i)

        stats.update()


# validate_hyperparameters()


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

    optimizer = torch.optim.Adam(
        network.parameters(), lr=10 * 1e-4, amsgrad=True, weight_decay=0.3)

    if load == True:
        optimizer.load_state_dict(torch.load("optimizer1"))

    stats = learningStats()

    for epoch in range(num_epochs):
        time_start = datetime.now()
        network.train()

        for i, (sample, label) in enumerate(loaded_train):
            sample.to(device)

            desired = torch.zeros((10, 1, 1, 1))
            desired[label, ...] = 1

            output = network(sample)

            loss = criterion.numSpikes(output, desired)

            stats.training.correctSamples += torch.sum(
                snn.predict.getClass(output) == label).data.item()
            stats.training.numSamples += len(label)

            if i % 50 == 0:
                logging.debug("____________________________________")
                logging.debug(f"The desired class is {int(label)} ")
                spikes = image_to_spike_tensor(sample, torch.zeros(
                    (1, 3, 32, 32, network.nTimeBins)), 1)
                res_psp = network.slayer.psp(network.fc1(spikes))
                res = network.slayer.spike(res_psp)
                res2 = network.slayer.psp(network.fc2(res))
                res2_spikes = network.slayer.spike(res2)

                for j in range(network.fc1.weight.shape[0]):
                    logging.debug(
                        f"The number of times neuron {j} fired from the 1st layer is {torch.sum(res[0][j][0][0])}")
                    avg = float(
                        torch.sum(res_psp[0][j][0][0])) / float(torch.numel(res_psp[0][j][0][0]))
                    logging.debug(
                        f"The average membrane potential for neuron {j} from the 1st layer is {avg}")
                
                logging.debug("____________________________________")

                for j in range(10):
                    avg = float(torch.sum(res2[0][j][0][0])) / \
                        float(torch.numel(res2[0][j][0][0]))
                    logging.debug(
                        f"The average membrane potential for neuron {j} in 2nd layer is {avg}")

                logging.debug("____________________________________")
                for j in range(10):
                    logging.debug(
                        f"The number of times neuron {j} fired from the 1st layer is {torch.sum(res2_spikes[0][j][0][0])}")

                logging.debug("____________________________________")

            stats.training.lossSum += loss.cpu().data.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if i % 10 == 0:
                stats.print(epoch, i, (datetime.now() -
                            time_start).total_seconds())

        torch.save(network.state_dict(), "network" + epoch)
        torch.save(optimizer.state_dict(), "optimizer" + epoch)

        logging.info("Starting the testing")

        network.eval()

        for i, (input, label) in enumerate(loaded_test, 0):
            input = input.to(device)

            output = network.forward(input)

            stats.testing.correctSamples += torch.sum(
                    snn.predict.getClass(output) == label).data.item()
            stats.testing.numSamples += len(label)

            loss = criterion.numSpikes(output, label)
            stats.testing.lossSum += loss.cpu().data.item()
            if i % 100 == 0:
                stats.print(epoch, i)

        stats.update()
