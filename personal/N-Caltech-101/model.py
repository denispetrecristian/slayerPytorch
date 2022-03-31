import slayerSNN as snn
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import learningstats


logging.basicConfig(filename='dvscifar10.log', level=logging.DEBUG)

netParams = snn.params("ncaltch101.yaml")
device = torch.device("cuda")


class NCaltech(Dataset):
    def __init__(self, datasetPath, sampleFile, samplingTime, sampleLength):
        self.path = datasetPath
        self.samples = np.genfromtxt(sampleFile, dtype=str)
        self.samplingTime = samplingTime
        self.nTimeBins = int(sampleLength / samplingTime)

    def __getitem__(self, index):
        inputPath = self.samples[index, 0]
        inputLabel = self.samples[index, 1]

        desired = torch.zeros((100, 1, 1, 1))
        desired[int(inputLabel), ...] = 1

        spikes = snn.io.read2Dspikes(
            self.path + inputPath).toSpikeTensor(torch.zeros((2, 200, 156, self.nTimeBins)))

        return spikes, inputLabel, desired

    def __len__(self):
        return len(self.samples)


class Network(torch.nn.Module):
    def __init__(self, netParams):
        super(Network, self).__init__()
        self.slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.fc1 = self.slayer.dense((156, 200,2), 800)
        self.fc2 = self.slayer.dense(800, 100)

    def forward(self, input):
        x = self.slayer.spike(self.slayer.psp(self.fc1(input)))
        x = self.slayer.spike(self.slayer.psp(self.fc2(x)))

        return x


def main():
    NCaltechTrain = NCaltech(netParams['training']['path']['in'], netParams['training']
                             ['path']['train'], netParams['simulation']['Ts'], netParams['simulation']['tSample'])

    NCaltechTest = NCaltech(netParams['training']['path']['in'], netParams['training']
                             ['path']['test'], netParams['simulation']['Ts'], netParams['simulation']['tSample'])

    loadedTrain = DataLoader(NCaltechTrain, shuffle=True)
    loadedTest = DataLoader(NCaltechTest, shuffle=True)

    for i in range(0):
        (sample,label,desired) = NCaltechTrain[i]
        snn.io.showTD(snn.io.spikeArrayToEvent(
            sample.reshape((2, 200, 156, -1)).cpu().data.numpy()))

    network = Network(netParams).to(device)
    criterion = snn.loss(netParams).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=0.5, amsgrad=True)

    stats = learningstats.learningStats()

    for epoch in range(10):
        tSt = datetime.now()
        
        for i, (input, label, target) in enumerate(loadedTrain, 0):
     
            input  = input.to(device)
            target = target.to(device) 
            label = torch.tensor([int(label[0])])
            output = network.forward(input)
            
            stats.training.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
            stats.training.numSamples     += len(label)
            
            loss = criterion.numSpikes(output, target)
            
            optimizer.zero_grad()
            
            loss.backward()
           
            optimizer.step()

            stats.training.lossSum += loss.cpu().data.item()

            # Display training stats.
            if i%10 == 0:   stats.print(epoch, i, (datetime.now() - tSt).total_seconds())


        torch.save(network.state_dict(), "network_caltech" + str(epoch))
        torch.save(optimizer.state_dict(), "optimizer_caltech" + str(epoch))               


        for i, (input, label, target) in enumerate(loadedTest, 0):
            input  = input.to(device)
            target = target.to(device)
            label = torch.tensor([int(label[0])]) 
            
            output = network.forward(input)

            stats.testing.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
            stats.testing.numSamples     += len(label)

            loss = criterion.numSpikes(output, target)
            stats.testing.lossSum += loss.cpu().data.item()
            if i%10 == 0:   stats.print(epoch, i)
        
        # Update stats.
        stats.save()
        stats.update()


if __name__ == "__main__":
    main()
