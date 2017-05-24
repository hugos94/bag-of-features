import numpy as np
import cv2
from sklearn.utils import shuffle
from skimage import io
from skimage import exposure
import os

def preprocess_dataset(X, y = None):
    #Convert to grayscale, e.g. single Y channel
    X = 0.299 * X[:, :, :, 0] + 0.587 * X[:, :, :, 1] + 0.114 * X[:, :, :, 2]
    #Scale features to be in [0, 1]
    X = (X / 255.).astype(np.float32)

    # Apply localized histogram localization
    for i in range(X.shape[0]):
        X[i] = exposure.equalize_adapthist(X[i])

    if y is not None:
        # Convert to one-hot encoding. Convert back with `y = y.nonzero()[1]`
        y = np.eye(43)[y]
        # Shuffle the data
        X, y = shuffle(X, y)

    # Add a single grayscale channel
    X = X.reshape(X.shape + (1,))
    return X, y

# Load images from .png files to `X_custom` NumPy array
#X_custom = np.empty([0, 32, 32, 3], dtype = np.int32)
#for i in range(38):
#    image = io.imread('../wenderDatabase/training/P/' + "image{}".format(i + 1) + '.jpg')
#    X_custom = np.append(X_custom, [image[:, :, :3]], axis = 0)

#X_custom, _ = preprocess_dataset(X_custom)

# for i in range(38):# Prepare original and preprocessed images
#     original = io.imread(os.getcwd() + '../wenderDatabase/training/P/' + "image{}".format(index + 1) + '.png')
#     preprocessed = X_custom[index].reshape(32, 32)
#
#     # Prepare the grid
#     pyplot.figure(figsize = (6, 2))
#     gridspec.GridSpec(2, 2)
#
#     # Plot original image
#     pyplot.subplot2grid((2, 2), (0, 0), colspan=1, rowspan=1)
#     pyplot.imshow(original)
#     pyplot.axis('off')
#
#     # Plot preprocessed image
#     pyplot.subplot2grid((2, 2), (1, 0), colspan=1, rowspan=1)
#     pyplot.imshow(preprocessed, cmap='gray')
#     pyplot.axis('off')
#
#     pyplot.show()


sift = cv2.xfeatures2d.SIFT_create()
X = cv2.imread("../germanDatabase/training/00000/00000_00028.ppm")
cv2.imwrite('original.ppm',X)
X = cv2.cvtColor(X,cv2.COLOR_RGB2GRAY)
cv2.imwrite('gray.ppm',X)
X = (X / 255.).astype(np.float32)
cv2.imwrite('normalized.ppm',X)
X = exposure.equalize_hist(X)
cv2.imwrite('globalequalization.ppm',X)
# for i in range(X.shape[0]):
#     X[i] = exposure.equalize_adapthist(X[i])
# print(X)

#(kpts, des) = sift.detectAndCompute(X, None)
#print("Keypoints: {}".format(kpts))
#print("Features: {}".format(des))
#X = 0.299 * X[:, :, :, 0] + 0.587 * X[:, :, :, 1] + 0.114 * X[:, :, :, 2]
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow("Image", X)
cv2.waitKey(3000)

# cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
# cv2.imshow("Image", X.reshape(32,32))
# cv2.waitKey(3000)
