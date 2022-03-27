'''
Script should be run from the dataset\cy6cvx3ryv-1 folder
'''


import os

# print(os.listdir("Caltech101_annotations"))
location = "Caltech101"
categories = os.listdir(location)

i = 0
for cat in categories:
    samples_number = os.listdir(location + "/" + cat)
    i+=1

# Split the dataset into training and testing
samples_train = []
samples_test = []
label = 0
for cat in categories:
    samples = os.listdir(location + "/" + cat)
    train_len = len(samples) / 4 * 3
    test_len = len(samples) - train_len
    for i in range(len(samples)):
        if i <= train_len:
            path = location + "/" + cat + "/" + samples[i]
            samples_train.append((path,label))
        else:
            path = location + "/" + cat + "/" + samples[i]
            samples_test.append((path,label))
    
    label+=1

with open("trainSamples.txt", "w") as f:
    for (sample,label) in samples_train:
        f.write(sample)
        f.write(" ")
        f.write(str(label) + "\n")


with open("testSamples.txt", "w") as f:
    for (sample,label) in samples_test:
        f.write(sample)
        f.write(" ")
        f.write(str(label) + "\n")

