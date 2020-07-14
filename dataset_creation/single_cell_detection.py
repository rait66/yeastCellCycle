import cv2
import numpy as np
import matplotlib.pyplot as plt
#import scipy.misc
import imageio
from idx_tools import Idx

import os

def convert_image(img, blur=3):
    # Convert to grayscale
    conv_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding to binarize the image
    #conv_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)
    _, conv_img = cv2.threshold(conv_img, 30, 255, cv2.THRESH_BINARY)
    # Blur the image to reduce noise
    conv_img = cv2.medianBlur(conv_img, blur)

    return conv_img

# Read image
img = cv2.imread('./images/2020-06-22-ML-21.jpg')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#_, conv_image=cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)


conv_img = convert_image(img)

# Show the effects of each processing stage
fig, ax = plt.subplots(1, 2, figsize=(20, 15))

cmaps = ['gray', 'gray']
titles = ['Original', 'Converted']
data = [img,conv_img]
for i in range(2):
    ax[i].imshow(data[i], cmap=cmaps[i])
    ax[i].set_title(titles[i], fontsize=23)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
"""ax[0].imshow(data[0], cmap='gray')
ax[1].imshow(conv_image, cmap='gray')"""
plt.show()

def extract_char(conv_img):
    # Find contours
    ctrs,_= cv2.findContours(conv_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return ctrs

def save_char(ctrs, img, lower=600, upper=20000, path='./char'):

    # Create the target folder for saving the extracted images
    if not os.path.isdir(path):
        os.mkdir(path)

    # Convert original image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Number of images already in the target folder
    n = len(os.listdir('./char')) + 1

    # Number of potential characters found in the image
    n_char = np.shape(ctrs)[0]

    # Go through each potential character
    for i in range(n_char):

        # Get coordinates of the potential character
        x, y, w, h = cv2.boundingRect(ctrs[i])

        # Test if the number of pixels in the bounding box is reasonable
        if (w * h) > lower and (w * h) < upper:

            # Draw the bounding box in the original image
            result = cv2.rectangle(img, (x, y), ( x + w, y + h ), (0, 255, 0), 2)

            # Extract the character and save it as a .jpeg
            roi = gray[y:y+h, x:x+w]
            imageio.imwrite('{}/char_{}.jpg'.format(path, n), roi)

            # Increase counter
            n += 1

    # Return image with all bounding boxes
    return result

# List of all images to create the first training data
image_files = ['./images/2020-06-22-ML-21.jpg','./images/2020-06-22-ML-22.jpg','./images/2020-06-22-ML-23.jpg','./images/2020-06-22-ML-24.jpg','./images/2020-06-22-ML-25.jpg','./images/2020-06-22-ML-26.jpg','./images/2020-06-22-ML-27.jpg','./images/2020-06-22-ML-28.jpg','./images/2020-06-22-ML-29.jpg','./images/2020-06-22-ML-30.jpg','./images/2020-06-22-ML-31.jpg','./images/2020-06-22-ML-32.jpg','./images/2020-06-22-ML-33.jpg','./images/2020-06-22-ML-34.jpg']

# Go through all files and extract the characters
for file in image_files:

    # Read image
    img = cv2.imread(file)

    # Convert the image (gray/thresholded/blured)
    conv_img = convert_image(img)

    # Find and sort the contours
    ctrs = extract_char(conv_img)

    # Save the result
    result = save_char(ctrs, img)


# Plot the result of the last image
fig, ax = plt.subplots(1, 1, figsize=(18, 32))

ax.imshow(result)
ax.set_xlim([0, 600])
ax.set_ylim([0, 600])
ax.set_xticks([])
ax.set_yticks([])

plt.show()

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Path to the individual letters
data_path = './char/'

# Target image size
convSize = [80, 80]

# File names of all sample images
files = [data_path + x for x in os.listdir(data_path)]

# Resize images and append to a numpy array
images = []
for file in files:
    img = cv2.imread(file)[:, :, 0]
    img = cv2.resize(img, (convSize[0], convSize[1]))
    img = img.reshape(convSize[0] * convSize[1])
    images.append(img)

images = np.array(images, dtype='float64')

# Apply StandardScaler on the letter data
scaler = StandardScaler()
scaler.fit(images)
scaled = scaler.transform(images)

# Calculate the first 25 principal componnents
pca = PCA(n_components=25)
pca.fit(scaled)
pca_img = pca.transform(scaled)


# Use K-Means clustering to group the data into 100 clusters
nClusters = 10
kmeans = KMeans(n_clusters=nClusters, random_state=0).fit(pca_img)


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')


ax.scatter(pca_img[:, 0], pca_img[:, 1], pca_img[:, 2], c=kmeans.labels_)

ax.set_xlabel('component 1', fontsize=18)
ax.set_ylabel('component 2', fontsize=18)
ax.set_zlabel('component 3', fontsize=18)

plt.show()



path = './clustered'

if not os.path.isdir(path):
    os.mkdir(path)

n = 0
for i in range(kmeans.labels_.max()):

    cluster_path = '{}/{}'.format(path, i)

    if not os.path.isdir(cluster_path):
        os.mkdir(cluster_path)

    tmp = images[kmeans.labels_ == kmeans.labels_[i]]

    for j in range(np.shape(tmp)[0]):
        tmpImg = np.reshape(tmp[j], convSize).astype(np.uint8)
        imageio.imwrite('{}/{}.jpg'.format(cluster_path, n), tmpImg)
        n += 1

"""# Delete the un-clustered data
[os.remove(data_path + x) for x in os.listdir(data_path)]
os.rmdir(data_path)"""
