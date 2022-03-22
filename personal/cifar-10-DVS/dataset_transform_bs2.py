from ast import Str
import slayerSNN as snn
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from AedatLegacy import LegacyAedatFile
import os
from learningstats import learningStats
import logging
from datetime import datetime
import zipfile
from graphviz import Digraph
import torch
from torch.autograd import Variable, Function
import aedat

netParams = snn.params("network.yaml")

categories = ["airplane", "automobile", "bird", "cat",
              "deer", "dog", "frog", "horse", "ship", "truck"]


def parse_sample_name(sample: Str):
    '''
        Function used to get the class of the sample
    '''
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

        x = []
        y = []
        p = []
        t = []

        desired = torch.zeros([10, 1, 1, 1])

        desired[category, ...] = 1

        sample_index = "dataset4/" + sample_index
        decoder = aedat.Decoder(sample_index + "4")

        for packet in decoder:
            if 'events' in packet:
                for i in range(len(packet['events'])):
                    if (packet['events'][i][0] / 1000) < self.nTimeBins:
                        x.append(packet['events'][i][1])
                        y.append(packet['events'][i][2])
                        t.append(packet['events'][i][0] / 1000)
                        p.append(packet['events'][i][3])

        decoder = None

        x = np.array(x)
        y = np.array(y)
        p = np.array(p)
        t = np.array(t)

        event = snn.io.event(x,y,p,t)
        name = self.samples[index].replace(".aedat", ".bs2")
        snn.io.encode2Dspikes("datasetbs2/" + name, event)
        # print(categories[category])
        # print(self.samples[index])
        # return snn.io.event(x, y, p, t).toSpikeTensor(torch.empty((2, 128, 128, self.nTimeBins))), category, desired
        return 1,2,3

    def __len__(self):
        return len(self.samples)


def transform():
    dataset_train = Cifar10DVS(netParams['training']['path']['in'], netParams['training']
                               ['path']['train'], netParams['simulation']['Ts'], netParams['simulation']['tSample'])

    dataset_test = Cifar10DVS(netParams['training']['path']['in'], netParams['training']
                              ['path']['test'], netParams['simulation']['Ts'], netParams['simulation']['tSample'])

    loaded_train = DataLoader(
        dataset_train, batch_size=1, num_workers=0, shuffle=False)
    loaded_test = DataLoader(
        dataset_test, batch_size=1, num_workers=0, shuffle=False)


    for i, (sample, label, desired) in enumerate(loaded_train):
        print(i)

    for i, (sample, label, desired) in enumerate(loaded_test):
        print(i)

transform()