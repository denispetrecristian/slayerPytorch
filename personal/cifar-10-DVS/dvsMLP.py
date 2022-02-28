from ast import Str
import slayerSNN as snn
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from AedatLegacy import LegacyAedatFile
import os
import logging

logging.basicConfig(filename='rccifar10.log',
                    encoding='utf-8', level=logging.DEBUG)


categories = ["airplane", "automobile", "bird", "cat",
              "deer", "dog", "frog", "horse", "ship", "truck"]


netParams = snn.params("network.yaml")
device = torch.device("cuda")

NUM_EPOCHS = 30
LEARNING_RATE = 0.01
BATCH_SIZE = 1
NUM_WORKERS = 0


def parse_sample_name(sample: Str):
    '''
        Function used to get the class of the sample
    '''
    for i in range(len(categories)):
        if categories[i] in sample:
            return i

    raise Exception(
        f"Value exception: Sample {sample} is not following the right naming format. Should contain one of the Cifar10 Categories ")


class Cifar10DVS(Dataset):
    def __init__(self, datasetPath, samplePath, samplingTime, sampleLentgh):
        self.path = datasetPath
        self.samples = np.loadtxt(samplePath)
        self.samplingTime = samplingTime
        self.nTimeBins = int(sampleLentgh / samplingTime)

    def __getitem__(self, index):
        sample_index = self.samples[index]
        category = torch.tensor(parse_sample_name(sample_index))

        x = np.array([])
        y = np.array([])
        p = np.array([])
        t = np.array([])

        desired = torch.empty([10, 1, 1, 1])
        desired[category, ...] = 1

        with LegacyAedatFile(sample_index) as f:
            for event in f:
                x = np.insert(x, event.x)
                y = np.insert(y, event.y)
                p = np.insert(p, event.polarity)
                t = np.insert(t, event.timestamp)

            logging.debug("Read the sample")

        return snn.io.event(x, y, p, t).toSpikeTensor(torch.empty([p.size, x.size, y.size, t.size])), category, desired

    def __len__(self):
        return len(self.samples)


class Network(torch.nn.Module):
    def __init__(self, netParams):
        self.slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.fc1 = self.slayer.dense((128, 128), 240)
        self.fc2 = self.slayer.dense(240, 10)

    def forward(self, input):
        layer1 = self.slayer.spike(self.slayer.psp(self.fc1(input)))
        layer2 = self.slayer.spike(self.slayer.psp(self.fc2(layer1)))

        return layer2


def overfit_single_batch():
    # Load datasets
    dataset_train = Cifar10DVS(netParams['training']['path']['in'], netParams['training']
                               ['path']['train'], netParams['neuron']['Ts'], netParams['neuron']['tSample'])

    dataset_test = Cifar10DVS(netParams['training']['path']['in'], netParams['training']
                              ['path']['test'], netParams['neuron']['Ts'], netParams['neuron']['tSample'])

    loaded_train = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    loaded_test = DataLoader(
        dataset_test, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    model = Network(netParams).to(device)
    criterion = snn.loss(netParams).to(device)
    optimizer = torch.optim.Adam(model.parameters, lr=LEARNING_RATE)




def main():
    dataset_train = Cifar10DVS(netParams['training']['path']['in'], netParams['training']
                               ['path']['train'], netParams['neuron']['Ts'], netParams['neuron']['tSample'])

    dataset_test = Cifar10DVS(netParams['training']['path']['in'], netParams['training']
                              ['path']['test'], netParams['neuron']['Ts'], netParams['neuron']['tSample'])

    loaded_train = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    loaded_test = DataLoader(
        dataset_test, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    model = Network(netParams).to(device)
    criterion = snn.loss(netParams).to(device)
    optimizer = torch.optim.Adam(model.parameters, lr=LEARNING_RATE)

    for epoch in NUM_EPOCHS:
        pass


if __name__ == "__main__":
    main()
