import numpy as np
from idx_tools import Idx
import os

# Where is the sorted data located?
dataset_path = './data/'
if not os.path.isdir(dataset_path):
    os.mkdir(dataset_path)
# Where should the converted data be stored?
dest_folder = './datasetTMP/'

# What is the ratio of train vs. test data?
train_percent = 0.7

# Subfolders for structuring the sorted data
train_path = dataset_path + 'train/'
test_path = dataset_path + 'test/'

# Find all folders in the sorted data
folders = os.listdir(dataset_path)

# Create folders
if not os.path.isdir(train_path):
    os.mkdir(train_path)
    os.mkdir(test_path)

if not os.path.isdir(dest_folder):
    os.mkdir(dest_folder)

# Go through all folders in the sorted data and split into train and test
for char in folders:
    char_path = dataset_path + char + '/'
    train_folder = train_path + char + '/'
    test_folder = test_path + char + '/'

    samples = os.listdir(char_path)
    n_samples = len(samples)
    n_train = round(train_percent * n_samples)

    sel = np.arange(n_samples)
    np.random.shuffle(sel)

    idx_train = sel[0:n_train]
    idx_test = sel[n_train:]

    if not os.path.isdir(train_folder):
        os.mkdir(train_folder)

    if not os.path.isdir(test_folder):
        os.mkdir(test_folder)

    [os.rename(char_path + samples[x], train_folder + samples[x]) for x in idx_train]
    [os.rename(char_path + samples[x], test_folder + samples[x]) for x in idx_test]

    os.rmdir(char_path)

# Convert data to idx format
Idx.save_idx(train_path)
Idx.save_idx(test_path)

# Move converted dataset to target folder
os.rename(train_path + 'images.idx3-ubyte', dest_folder + 'train-images.idx3-ubyte')
os.rename(train_path + 'labels.idx3-ubyte', dest_folder + 'train-labels.idx3-ubyte')
os.rename(test_path + 'images.idx3-ubyte', dest_folder + 'test-images.idx3-ubyte')
os.rename(test_path + 'labels.idx3-ubyte', dest_folder + 'test-labels.idx3-ubyte')

chars = os.listdir('./data/train')

labelFile = open('{}/labels.txt'.format(dest_folder), "w")
for char in chars:

    if char.endswith('_'):
        char = char[:-1].upper()

    # write line to output file
    labelFile.write(char)
    labelFile.write("\n")
labelFile.close()
