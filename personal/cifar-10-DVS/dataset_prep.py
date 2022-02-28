import os
import random

all_files = []
dir = os.listdir("dataset")
for entry in dir:
    folder_path = os.getcwd() + "/" + "dataset" + "/" + str(entry)
    if os.path.isdir(folder_path) == False:
        print("is not dir")
        continue
    file_loc = os.listdir("dataset/" + str(entry))
    for file in file_loc:
        all_files.append(str(entry) + "/" + str(file))


for file in all_files:
    print(file)

# Split the data in training and testing
random.shuffle(all_files)

print(len(all_files))

train_set = all_files[0 : 9000]
test_set = all_files[9000 : 10000]

with open("dataset/train9000.txt", 'w') as f:
    for file in train_set:
        f.write(file)
        f.write("\n")

with open("dataset/test1000.txt", 'w') as f:
    for file in test_set:
        f.write(file)
        f.write("\n")
