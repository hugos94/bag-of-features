import cv2, os, time, imutils
import argparse as ap
import numpy as np
import logging as log
from multiprocessing import Pool
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

def detectAndCompute(image_paths):
    des_list = []
    sift = cv2.xfeatures2d.SIFT_create()
    for image_path in image_paths:
        im = cv2.imread(image_path)
        (kpts, des) = sift.detectAndCompute(im, None)
        des_list.append((image_path,des))
    return des_list

if __name__ == "__main__":
    # Get the path of the training set
    parser = ap.ArgumentParser()
    parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
    parser.add_argument("-c", "--classifierModelFile", help="Classifier Model File", required="True")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
        log.info("Verbose output.")
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    log.info("Algorithm execution time count started")
    start_time = time.time()

    # Get the training classes names and store them in a list
    log.info("Getting the training classes names and store them in a list")
    train_path = args.trainingSet
    training_names = os.listdir(train_path)

    # Get all the path to the images and save them in a list
    # image_paths and the corresponding label in image_paths
    log.info("Getting all the path to the images and save them in a list")
    image_paths = []
    image_classes = []
    class_id = 0
    for training_name in training_names:
        dir = os.path.join(train_path, training_name) # Training directories
        class_path = imutils.imlist(dir)
        image_paths += class_path
        image_classes += [class_id]*len(class_path)
        class_id += 1

    # Create feature extraction and keypoint detector objects
    log.info("Create feature extraction and keypoint detector objects")
    cpus = os.cpu_count()
    path_size = len(image_paths)
    path_lists_size = int(len(image_paths)/cpus)
    log.info("Dividing feature extraction between {} cpus".format(cpus))

    image_paths_parts = [image_paths[i:i + path_lists_size] for i in range(0, path_size, path_lists_size)]

    pool = Pool(processes=cpus)

    # List where all the descriptors are stored
    log.info("Detecting points and extracting features")
    features = pool.map(detectAndCompute, (image_paths_parts))

    # Stack all the descriptors vertically in a numpy array
    log.info("Stack all the descriptors vertically in a numpy array")
    descriptors = features[0].pop(0)[1]
    for feature in features:
        for _, descriptor in feature:
                descriptors = np.vstack((descriptors, descriptor))

    # Perform k-means clustering
    log.info("Perform k-means clustering")
    k = 100
    voc, _ = kmeans(descriptors, k, 1)

    # Calculate the histogram of features
    log.info("Calculate the histogram of features")
    im_features = np.zeros((len(image_paths), k), "float32")
    i = 0
    for feature in features:
        for image_path, descriptor in feature:
            words, _ = vq(descriptor, voc)
            for w in words:
                im_features[i][w] +=1
            i += 1

    # Perform Tf-Idf vectorization
    log.info("Perform Tf-Idf vectorization")
    nbr_ocurrences = np.sum( (im_features > 0) * 1, axis = 0)
    idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_ocurrences + 1)), 'float32')

    # Scaling the words
    log.info("Scaling the words")
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)

    # Train the Linear SVM
    log.info("Train the Linear SVM")
    clf = LinearSVC()
    clf.fit(im_features, np.array(image_classes))

    # Save the SVM
    log.info("Saving SVM")
    bof = args.classifierModelFile + ".pkl"
    joblib.dump((clf, training_names, stdSlr, k, voc), bof, compress = 3)

    # Print algorithm execution time in seconds
    log.info("Algorithm execution time count ended")
    log.info("--- %s seconds ---" % (time.time() - start_time))
