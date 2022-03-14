import numpy as np
import matplotlib.pyplot as plt

n1 = 410
n2 = 10

psp_values_layer1 = []
psp_values_layer2 = []
spike_numbers_layer1 = []
spike_numbers_layer2 = []

for i in range(n1):
    psp_values_layer1.append([])
    spike_numbers_layer1.append([])

for i in range(n2):
    psp_values_layer2.append([])
    spike_numbers_layer2.append([])


def parse_logging_file(path: str, network: str):
    with open(path) as f:
        lines = f.readlines()
        lines = lines[4:]
        for line in lines:
            words = line.split()

            if "potential" in words:
                neuron, layer, potential = add_to_potential(words)
                if layer == 1:
                    psp_values_layer1[neuron].append(potential)
                if layer == 2:
                    psp_values_layer2[neuron].append(potential)

            if "fired" in words:
                neuron, layer, spikes = add_to_spikes(words)
                if layer == 1:
                    spike_numbers_layer1[neuron].append(spikes)
                if layer == 2:
                    spike_numbers_layer2[neuron].append(spikes)


def add_to_potential(log_line: list):
    # Get the neuron number
    index = 0
    layer = 0

    for i in range(len(log_line)):
        if log_line == "neuron":
            index = int(log_line[i+1])

    if "1st" in log_line:
        layer = 1
    if "2nd" in log_line:
        layer = 2

    if layer == 0:
        raise Exception("The line did not have a layer specified")

    potential = float(log_line[-1])

    return index, layer, potential


def add_to_spikes(log_line: list):
    # Get the neuron number
    index = 0
    layer = 0
    for i in range(len(log_line)):
        if log_line == "neuron":
            index = int(log_line[i+1])

    if "1st" in log_line:
        layer = 1
    if "2nd" in log_line:
        layer = 2

    if layer == 0:
        raise Exception("The line did not have a layer specified")

    spikes = float(log_line[-1])

    return index, layer, spikes


def draw_data(data):
    data = np.array(data)
    for i in range(len(data)):
        plt.plot(np.arange(0, len(data[i])), data[i])
    plt.show()


def check_non_firing_neuron(spike_numbers):
    for i in range(n1):
        sum = 0
        for times in spike_numbers[i]:
            sum+=times
        
        if sum == 0:
            print(f"Neuron {i} did not fire over the whole simulation")
        
        # if sum < 100:
        #     print(f"Neuron {i} only fired less than a hundred times in the simulation")

parse_logging_file("rccifar10.log", "")
draw_data(psp_values_layer1)
draw_data(spike_numbers_layer1)
check_non_firing_neuron(spike_numbers_layer1)