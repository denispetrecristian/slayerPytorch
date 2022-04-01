# Explanation of folder

The project contains models for 5 datasets:

- N-caltech101 (neuromorphic version of Caltech101 caught using a Dynamic Vision Sensor)
- CIFAR10-DVS (neuromorphic version of CIFAR10 caught using a Dynamic Vision Sensor)
- Fashion MNIST
- CIFAR10
- CIFAR100

The project also contains changes to the source code of the original Slayer algorithm
by adding CUDA C/C++ methods to encode a non-neuromorphic dataset to one that can be
used by the library. Further changes have been made to the source code such that
different aproximations can be used for the derivative of the Spiking function.

# CIFAR 10

The `testCuda` file was only used when testing the encoding mechanisms.

The `spike_analysis.py` file is a script I have used to vizualize information
produced by the models. The only valid input is a log file of the form that has
been used in the models.

`*.yaml` are files that describe the hyperparameters for the simulations

`poissonCifar10.py` is an attempt at the model for CIFAR10 using poisson
encoding. NOT IN USE

`learningstats.py` is a recurring file, part of the original library, used to
gather information about the simulations.

`interpolateCifar10.py` is the model for CIFAR10 using interpolation encoding

`gradCheck.py` is a script created by a Meta employee, published on their forums,
that helps users vizualize different information about the gradients. This script is
no longer used and calls to this throughtout the code are part of unreachable code.

`encoding.py` is another legacy file that was previously used to write the encoding
mechanisms.

`cifar10poisson` is the model for CIFAR10 using poisson encoding

`Aedat Legacy` is part of the dv-python library. This file does not correctly read
neuromorphic data.

# N-Caltech 101 #

`dataset_prep.py` is a script used to create text files with splitting of training and
testing data.

`learningstats.py` is the same file mentioned above

`model.py` is the model

# CIFAR10 DVS #

dataset and dataset4 are places you can store the dataset.

`AedatLegacy.py` is the same file mentioned above

`gradCheck.py` is the same file mentioned above

`dataset_transform_bs2.py` is the file used to convert smaple from .aedat4 to .bs2.
I have done this process because of the somewhat considerable increase in training speed
attributed to not having to read headers contain information about the frame, triggers and
so on.

`dataset_prep.py` is a script used to create text files with splitting of training and
testing data.

`learningstats.py` is the same file mentioned above

`dvsMLP.py` is the model


# FASHION MNIST and CIFAR 100 #

`learningstats.py` is the same file mentioned above

`model.py / fashionMNIST.py` is the model. This model is DOWNLOADING the dataset. 
Turn off from inside the code the variable download_value to FALSE in order to 
stop this dataset from being downloaded. These two models require the setting up
of command line flags

e.g. `python model.py -e poisson -m CNN` This runs the convolutional SNN with the poisson
encoding. The other 2 values for the `-e` flag are rate and interpolation. To run the MLP
you would have to use `-m MLP`.