from idx_tools import Idx
import matplotlib.pyplot as plt

# Read the data
mnist_data = Idx.load_idx('./datasetTMP/train-images.idx3-ubyte')

# Plot a random image
#plt.imshow(mnist_data[2034], cmap='gray')

#mnist_labels = Idx.load_labels('./convert_IDX/mnist/train-labels.idx1-ubyte')

plt.imshow(mnist_data[22],cmap='gray')
plt.show()