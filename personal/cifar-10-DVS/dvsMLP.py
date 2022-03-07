from ast import Str
import slayerSNN as snn
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from AedatLegacy import LegacyAedatFile
import os
from learningstats import learningStats
import logging
from datetime import datetime

logging.basicConfig(filename='dvscifar10.log',
                    encoding='utf-8', level=logging.DEBUG)


categories = ["airplane", "automobile", "bird", "cat",
              "deer", "dog", "frog", "horse", "ship", "truck"]


netParams = snn.params("network.yaml")
device = torch.device("cuda")

NUM_EPOCHS = 30
LEARNING_RATE = 0.01
BATCH_SIZE = 1
NUM_WORKERS = 0
SCALE_OVERFIT = 100


def parse_sample_name(sample: Str):
    '''
        Function used to get the class of the sample
    '''
    print(type(sample))
    print(type(categories[0]))
    for i in range(len(categories)):
        if str(categories[i]) in sample:
            return i

    raise Exception(
        f"Value exception: Sample {sample} is not following the right naming format. Should contain one of the Cifar10 Categories ")


class Cifar10DVS(Dataset):
    def __init__(self, datasetPath, samplePath, samplingTime, sampleLentgh):
        self.path = datasetPath
        self.samples = np.genfromtxt(samplePath, dtype=str)
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

        sample_index = "dataset/" + sample_index
        print(sample_index)

        with LegacyAedatFile(sample_index) as f:
            # print(len(iter(f)))
            for event in f:
                x = np.append(x, event.x)
                y = np.append(y, event.y)
                p = np.append(p, event.polarity)
                t = np.append(t, event.timestamp / 1000)

            logging.debug("Read the sample")

        print(p.size)
        print(x.size)
        print(y.size)
        print(t.size)

        return snn.io.event(x, y, p, t).toSpikeTensor(torch.empty([p.size, x.size, y.size, t.size])), category, desired

    def __len__(self):
        return len(self.samples)


class Network(torch.nn.Module):
    def __init__(self, netParams):
        super(Network,self).__init__()
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
                               ['path']['train'], netParams['simulation']['Ts'], netParams['simulation']['tSample'])

    dataset_test = Cifar10DVS(netParams['training']['path']['in'], netParams['training']
                              ['path']['test'], netParams['simulation']['Ts'], netParams['simulation']['tSample'])

    loaded_train = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    loaded_test = DataLoader(
        dataset_test, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    print(len(dataset_test))
    print(len(dataset_train))

    model = Network(netParams).to(device)
    criterion = snn.loss(netParams).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    stats = learningStats()

    # Get single batch
    (sample, label, desired) = next(iter(loaded_train))
    sample = sample.to(device)
    label = label.to(device)
    desired = desired.to(device)

    for epoch in NUM_EPOCHS * SCALE_OVERFIT:
        output = model(sample)

        loss = criterion.numSpikes(output, desired)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())



def main():
    dataset_train = Cifar10DVS(netParams['training']['path']['in'], netParams['training']
                               ['path']['train'], netParams['simulation']['Ts'], netParams['simulation']['tSample'])

    dataset_test = Cifar10DVS(netParams['training']['path']['in'], netParams['training']
                              ['path']['test'], netParams['simulation']['Ts'], netParams['simulation']['tSample'])

    loaded_train = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    loaded_test = DataLoader(
        dataset_test, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    model = Network(netParams).to(device)
    criterion = snn.loss(netParams).to(device)
    optimizer = torch.optim.Adam(model.parameters, lr=LEARNING_RATE)

    stats = learningStats()

    for epoch in NUM_EPOCHS:
        tSt = datetime.now()
        for i, (sample, label, desired) in enumerate(loaded_train):
            sample = sample.to(device)
            label = label.to(device)
            desired = desired.to(device)

            output = model(sample)

            stats.training.correctSamples += torch.sum(
                snn.predict.getClass(output) == label).data.item()
            stats.training.numSamples += len(label)

            loss = criterion.numSpikes(output, desired)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            stats.training.lossSum += loss.cpu().data.item()

            if i % 10 == 0:
                stats.print(epoch, i, (datetime.now() - tSt).total_seconds())

        for i, (sample, label, desired) in enumerate(loaded_test):
            sample = sample.to(device)
            label = label.to(device)
            desired = desired.to(device)

            output = model(sample)

            stats.testing.correctSamples += torch.sum(
                snn.predict.getClass(output) == label).data.item()
            stats.testing.numSamples += len(label)

            loss = criterion.numSpikes(output, desired)

            stats.testing.lossSum += loss.cpu().data.item()
            if i % 10 == 0:
                stats.print(epoch, i)

        stats.update()


if __name__ == "__main__":
    overfit_single_batch()
