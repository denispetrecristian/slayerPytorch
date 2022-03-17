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
import zipfile
from graphviz import Digraph
import torch
from torch.autograd import Variable, Function


logging.basicConfig(filename='dvscifar10.log',
                    encoding='utf-8', level=logging.DEBUG)


categories = ["airplane", "automobile", "bird", "cat",
              "deer", "dog", "frog", "horse", "ship", "truck"]


netParams = snn.params("network.yaml")
device = torch.device("cuda")

NUM_EPOCHS = 30
LEARNING_RATE = 0.0002
BATCH_SIZE = 8
NUM_WORKERS = 0
SCALE_OVERFIT = 1


def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)

def register_hooks(var):
    fn_dict = {}
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output):
        if grad_output is None:
            return False
        return grad_output.isnan().any() or (grad_output.abs() >= 1e6).any()

    def make_dot():
        node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='12',
                        ranksep='0.1',
                        height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '('+(', ').join(map(str, size))+')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):  # if GradAccumulator
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                assert fn in fn_dict, fn
                fillcolor = 'white'
                if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = 'red'
                dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot


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

        desired = torch.empty([10, 1, 1, 1])
        desired[category, ...] = 1

        sample_index = "dataset/" + sample_index

        with LegacyAedatFile(sample_index) as f:
            # print(len(iter(f)))
            for event in f:
                x.append(event.x)
                y.append(event.y)
                p.append(event.polarity)
                t.append(int(event.timestamp/1000))

        x = np.array(x)
        y = np.array(y)
        p = np.array(p)
        t = np.array(t)

        return snn.io.event(x, y, p, t).toSpikeTensor(torch.empty((1, 128, 128, 1400))), category, desired

    def __len__(self):
        return len(self.samples)


class Network(torch.nn.Module):
    def __init__(self, netParams):
        super(Network, self).__init__()
        self.slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.fc1 = self.slayer.dense((128, 128), 410)
        self.fc2 = self.slayer.dense(410, 240)
        self.fc3 = self.slayer.dense(240, 10)

        # torch.nn.init.uniform(self.fc1.weight, 0, 1/30)
        # torch.nn.init.uniform(self.fc2.weight, 0, 1/30)

    def forward(self, input):
        layer1 = self.slayer.spike(self.slayer.psp(self.fc1(input)))
        layer2 = self.slayer.spike(self.slayer.psp(self.fc2(layer1)))
        layer3 = self.slayer.spike(self.slayer.psp(self.fc3(layer2)))

        return layer3


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

    model = Network(netParams).to(device)
    criterion = snn.loss(netParams).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    stats = learningStats()

    # Get single batch
    (sample, label, desired) = next(iter(loaded_train))
    sample = sample.to(device)
    label = label.to(device)
    desired = desired.to(device)

    for epoch in range(NUM_EPOCHS * SCALE_OVERFIT):
        output = model(sample)

        for j in range(10):
            print(
                f"The number of times neuron {j} fired is {torch.sum(output[0][j][0][0])}")

        loss = criterion.numSpikes(output, desired)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())


def main():

    # with zipfile.ZipFile("dataset.zip", "r") as zip_ref:
    #     zip_ref.extractall("dataset")

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
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.03)

    stats = learningStats()

    for epoch in range(NUM_EPOCHS):
        tSt = datetime.now()
        for i, (sample, label, desired) in enumerate(loaded_train):
            sample = sample.to(device)
            desired = desired.to(device)

            output = model(sample)

            if i % 30 == 0:
                for j in range(10):
                    logging.debug(
                        f"The number of times neuron {j} fired is {torch.sum(output[0][j][0][0])}")

                layer1_psp = model.slayer.psp(model.fc1(sample))
                layer1_spikes = model.slayer.spike(layer1_psp)

                layer2_psp = model.slayer.psp(model.fc2(layer1_spikes))

                for j in range(10):
                    avg = float(
                        torch.sum(layer2_psp[0][j][0][0])) / float(torch.numel(layer2_psp[0][j][0][0]))
                    logging.debug(
                        f"The average membrane potential for neuron {j} is {avg}")

            stats.training.correctSamples += torch.sum(
                snn.predict.getClass(output) == label).data.item()
            stats.training.numSamples += len(label)

            loss = criterion.numSpikes(output, desired)
            optimizer.zero_grad()

            if i % 50 == 0:
                get_dot = register_hooks(loss)
                loss.backward()
                dot = get_dot()
                dot.render("gradCifar10DVS", format="png")
            else:
                loss.backward()
            optimizer.step()

            stats.training.lossSum += loss.cpu().data.item()

            if i % 2 == 0:
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
    main()
