import os, cv2, time, numpy, imutils, argparse
import logging as log
from multiprocessing import Pool
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

def get_args():
    # Set and get script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trainingSetPath", help="Path to Training Set", required="True")
    parser.add_argument("-c", "--classifierModelFile", help="Classifier Model File", required="True")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    return parser.parse_args()

def detectAndCompute(image_paths):
    # Detect, compute and return all features found on images
    descriptions = []
    keypoint_detector = cv2.xfeatures2d.SIFT_create()
    keypoint_descriptor = cv2.xfeatures2d.SIFT_create()
    for image_path in image_paths:
        image = cv2.imread(image_path)
        keypoints = keypoint_detector.detect(image, None)
        (keypoints, description) = keypoint_descriptor.compute(image, keypoints)
        descriptions.append((image_path,description))
    return descriptions

def stack_descriptors(features):
    # Stack all the descriptors vertically in a numpy array
    descriptors = features.pop(0)[1]
    for _, descriptor in features:
        descriptors = numpy.concatenate((descriptors, descriptor), axis=0)
    return descriptors

if __name__ == "__main__":

    # Get arguments
    args = get_args()

    # If verbose argument is provided, allow logs to be printed
    if args.verbose:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    # Start counting execution time
    log.info("Starting counting execution time")
    start_time = time.time()

    # Get the training classes names and store them in a list
    log.info("Getting training classes names and store them in a list")
    try:
        training_names = os.listdir(args.trainingSetPath)
    except OSError:
        log.error("No such directory {}. Check if the directory exists".format(args.trainingSetPath))
        exit()

    # Get all paths to the images and save them in a list with image_paths and the corresponding label in image_paths
    log.info("Getting all paths to the images and save them in a list")
    image_paths = []
    image_classes = []
    class_id = 0
    for training_name in training_names:
        directory_name = os.path.join(args.trainingSetPath, training_name)
        class_path = imutils.imlist(directory_name)
        image_paths += class_path
        image_classes += [class_id]*len(class_path)
        class_id += 1

    # Get the amount of cpus
    cpus = os.cpu_count()

    # Take the set size
    set_size = len(image_paths)

    # Calculates the number of subsets required for the quantity of cpus
    subset_size = int(numpy.ceil(set_size/cpus))

    # Divide the set into subsets according to the quantity of cpus
    log.info("Dividing feature detection and extraction between {} processes".format(cpus))
    image_paths_parts = [image_paths[i:i + subset_size] for i in range(0, set_size, subset_size)]

    # Create feature extraction and keypoint detector objects
    log.info("Create feature extraction and keypoint detector objects")
    pool = Pool(processes=cpus)

    # Detecting points and extracting features
    log.info("Detecting points and extracting features")
    features = pool.map(detectAndCompute, (image_paths_parts))

    # Stack all the descriptors vertically in a numpy array using Pool
    log.info("Stack all descriptors vertically in a numpy array")
    descriptors = pool.map(stack_descriptors, (features))
    descriptors_result = descriptors.pop(0)
    for descriptor in descriptors:
        descriptors_result = numpy.concatenate((descriptors_result, descriptor), axis=0)

    # Perform k-means clustering
    log.info("Perform k-means clustering")
    k = 100
    voc, _ = kmeans(descriptors_result, k, 1)

    # Creating codebook
    log.info("Creating codebook")
    im_features = numpy.zeros((set_size, k), "float32")
    i = 0
    for feature in features:
        for _ , descriptor in feature:
            words, _ = vq(descriptor, voc)
            for w in words:
                im_features[i][w] +=1
            i += 1

    # Scaling the words
    log.info("Scaling words")
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)

    # Train the Linear SVM
    log.info("Train Linear SVM")
    clf = LinearSVC()
    clf.fit(im_features, numpy.array(image_classes))

    # Save the SVM
    log.info("Saving SVM")
    bof = args.classifierModelFile + ".pkl"
    joblib.dump((clf, training_names, stdSlr, k, voc), bof, compress = 3)

    # Stopping counting execution time
    log.info("Stopping counting execution time")
    log.info("--- %s seconds ---" % (time.time() - start_time))
