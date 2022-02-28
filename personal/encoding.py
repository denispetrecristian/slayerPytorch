from time import sleep
import slayerSNN as snn
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision
from AedatLegacy import LegacyAedatFile

class Event1D():
    def __init__(self, neuron_id, time):
        self.neuron_id = neuron_id
        self.time = time
        self.polarity = 1
    
    def to_TD_event(self):
        return snn.io.event(self.neuron_id, None, self.polarity, self.time)
        


class Event():
    def __init__(self, neuron_x_id, neuron_y_id, time) -> None:
        self.x = neuron_x_id
        self.y = neuron_y_id
        self.dim = 2
        self.time = time



class Encoding():

    def __init__(self, timelapse):
        self.timelapse = timelapse

    @staticmethod
    def calculate_spikes(pixel, timeBins):
        pixel = int(pixel)

        if pixel == 0:
            return np.array([])
        
                
 
    def _calculate_spike_times(self, number):

        if number == 0:
            return []

        spike_interval = self.timelapse/number
        spike_times = []

        for i in range(number):
            time = spike_interval * i
            spike_times.append(time)

        return spike_times

    @staticmethod
    def _from_2d_event_to_snn_event(event):
        return snn.io.event(event.x, event.y, 1, event.time)
    
    @staticmethod
    def _from_1d_event_to_snn_event(event):
        return snn.io.event(event.x, None, 1, event.time)

    def pixel_to_event_array(self,pixel, neuron_id):
        "Takes the value of the pixel and the position it is at and returns an array of TD events"

        pixel = int(pixel)
        times = self._calculate_spike_times(pixel)
        spike_train = []
        for time in times:
            event = Event1D(neuron_id, time)
            spike_train.append(event.to_TD_event())
        
        return spike_train
            

    @staticmethod
    def animate_event(event):
        return snn.io.animTD(Encoding._from_event_to_snn_event(event))

    @staticmethod
    def td_bi_event_to_bytes(TD):
        if TD.dim != 2: 	raise Exception('Expected Td dimension to be 2. It was: {}'.format(TD.dim))
        xEvent = np.round(TD.x).astype(int)
        yEvent = np.round(TD.y).astype(int)
        pEvent = np.round(TD.p).astype(int)
        tEvent = np.round(TD.t * 1000).astype(int)	# encode spike time in us
        # outputByteArray = bytearray(len(tEvent) * 5)
        outputByteArray = bytearray(5)
        outputByteArray[0::5] = np.uint8(xEvent).tobytes()
        outputByteArray[1::5] = np.uint8(yEvent).tobytes()
        outputByteArray[2::5] = np.uint8(((tEvent >> 16) & 0x7F) | (pEvent.astype(int) << 7) ).tobytes()
        outputByteArray[3::5] = np.uint8( (tEvent >> 8 ) & 0xFF ).tobytes()
        outputByteArray[4::5] = np.uint8(  tEvent & 0xFF ).tobytes()
        
        return outputByteArray
    
    @staticmethod
    def td_uni_event_to_bytes(TD):
        if TD.dim != 1: 	raise Exception('Expected Td dimension to be 1. It was: {}'.format(TD.dim))
        xEvent = np.round(TD.x).astype(int)
        pEvent = np.round(TD.p).astype(int)
        tEvent = np.round(TD.t * 1000).astype(int)	# encode spike time in us
        # outputByteArray = bytearray(len(tEvent) * 5)
        outputByteArray = bytearray(5)
        outputByteArray[0::5] = np.uint8( (xEvent >> 8) & 0xFF00 ).tobytes()
        outputByteArray[1::5] = np.uint8( (xEvent & 0xFF) ).tobytes()
        outputByteArray[2::5] = np.uint8(((tEvent >> 16) & 0x7F) | (pEvent.astype(int) << 7) ).tobytes()
        outputByteArray[3::5] = np.uint8( (tEvent >> 8 ) & 0xFF ).tobytes()
        outputByteArray[4::5] = np.uint8(  tEvent & 0xFF ).tobytes()

        return outputByteArray

    @staticmethod
    def write1dspike(TD, filename):
        if TD.dim != 1: 	raise Exception('Expected Td dimension to be 1. It was: {}'.format(TD.dim))
        xEvent = np.round(TD.x).astype(int)
        pEvent = np.round(TD.p).astype(int)
        tEvent = np.round(TD.t * 1000).astype(int)	# encode spike time in us
        # outputByteArray = bytearray(len(tEvent) * 5)
        outputByteArray = bytearray(5)
        outputByteArray[0::5] = np.uint8( (xEvent >> 8) & 0xFF00 ).tobytes()
        outputByteArray[1::5] = np.uint8( (xEvent & 0xFF) ).tobytes()
        outputByteArray[2::5] = np.uint8(((tEvent >> 16) & 0x7F) | (pEvent.astype(int) << 7) ).tobytes()
        outputByteArray[3::5] = np.uint8( (tEvent >> 8 ) & 0xFF ).tobytes()
        outputByteArray[4::5] = np.uint8(  tEvent & 0xFF ).tobytes()
        with open(filename, 'a+b') as outputFile:
            outputFile.write(outputByteArray)

    @staticmethod
    def write2Dspike(TD,filename):
        if TD.dim != 2: 	raise Exception('Expected Td dimension to be 2. It was: {}'.format(TD.dim))
        xEvent = np.round(TD.x).astype(int)
        yEvent = np.round(TD.y).astype(int)
        pEvent = np.round(TD.p).astype(int)
        tEvent = np.round(TD.t * 1000).astype(int)	# encode spike time in us
        # outputByteArray = bytearray(len(tEvent) * 5)
        outputByteArray = bytearray(5)
        outputByteArray[0::5] = np.uint8(xEvent).tobytes()
        outputByteArray[1::5] = np.uint8(yEvent).tobytes()
        outputByteArray[2::5] = np.uint8(((tEvent >> 16) & 0x7F) | (pEvent.astype(int) << 7) ).tobytes()
        outputByteArray[3::5] = np.uint8( (tEvent >> 8 ) & 0xFF ).tobytes()
        outputByteArray[4::5] = np.uint8(  tEvent & 0xFF ).tobytes()
        with open(filename, 'a+b') as outputFile:
            outputFile.write(outputByteArray)

    @staticmethod
    def from_dictionary_to_descriptor_file(sample_label:dict, filepath):
        with open(filepath, 'a') as f:
            for key in sample_label.keys():
                f.write(str(key) + "  " + str(sample_label[key]) + "\n")

    @staticmethod
    def readAedatFiles(filepath):
        with LegacyAedatFile(filepath) as f:
            i = 0
            for event in f:
                print(i)
                i +=1
                nmevent = Event(event.x, event.y, event.timestamp/1000)
                nmevent = Encoding._from_2d_event_to_snn_event(nmevent)
                Encoding.write2Dspike(nmevent, "nm0.bs2")
        

def main1():
    Cifar10Tensors = CIFAR10(root="", train=True, download=False, transform=torchvision.transforms.ToTensor())
    Cifar10TensorsData = DataLoader(Cifar10Tensors, shuffle=False, batch_size=1, num_workers=0)

    encoding = Encoding(2000)

    labels_seen = 0

    labels_samples = {}

    for i, (image,label) in enumerate(Cifar10TensorsData):
        X = torch.flatten(image)
        X = X * 255
        X = torch.floor(X)
        X = X / 4
        X = torch.floor(X)

        nr_sample = i
        neuron_id = 0
        print(f"The {i} sample out of 10000")

        labels_seen+=1

        if labels_seen > 900:
            if labels_seen > 6000:
                labels_seen = 0
            else:
                continue

        labels_samples[i] = label

        for pixel in X:
            # print("The pixel is: " + str(neuron_id))
            spike_train = encoding.pixel_to_event_array(pixel, neuron_id)
            for spike in spike_train:
                Encoding.write1dspike(spike, f'cifar-10-nm/{nr_sample}.bs1')

            neuron_id+=1

    Encoding.from_dictionary_to_descriptor_file(labels_samples, "cifar-10-nm/train10000.txt")

def main2():
    labels_samples = {}
    labels_samples[1] = 2
    labels_samples[2] = 3
    Encoding.from_dictionary_to_descriptor_file(labels_samples, "cifar-10-nm/train10000.txt")


def main3():
    pass



if __name__ == "__main__":
    # event = Event(1,1,10)
    # spike_event = Encoding._from_1d_event_to_snn_event(event)
    # Encoding.write1dspike(spike_event, "a.bs2")
    # spike_event = snn.io.read2Dspikes("a.bs2")
    # print(spike_event.y)
    # print(spike_event.t)
    main1()
