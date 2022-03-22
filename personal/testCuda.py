import slayerSNN as snn
import torchvision
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

netParams = snn.params("network.yaml")
device = torch.device("cuda")

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
        spikes = self.slayer.rateEncoding(input).cuda()
        print(spikes.shape)
        print("This worked?")
        layer1 = self.slayer.spike(self.slayer.psp(self.fc1(spikes)))
        layer2 = self.slayer.spike(self.slayer.psp(self.fc2(layer1)))

        return layer2

transformation = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), lambda x: x*255])

dataset_train = CIFAR10(root="", download=False,
                            transform=transformation, train=True)
dataset_test = CIFAR10(root="", download=False,
                           transform=transformation, train=False)

loaded_train = DataLoader(
        dataset_train, batch_size=1, num_workers=0, shuffle=False)
loaded_test = DataLoader(dataset_test, batch_size=1,
                             num_workers=0, shuffle=False)

network = Network().to(device)
criterion = snn.loss(netParams).to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
for i, (sample,label) in enumerate(loaded_train):
  sample = sample.to(device)
  output = network(sample)
  print(i)

