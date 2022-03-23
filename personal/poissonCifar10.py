from turtle import forward
import torch
import slayerSNN as snn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
import learningstats
import torchvision

transformation = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), lambda x: x*255])

netParams = snn.params("networkPoisson.yaml")


class Network(torch.nn.Module):
    def __init__(self):
        super(Network).__init__()
        self.slayer = snn.layer(netParams['neuron'], netParams['simulation'])

    def forward(input):
        pass