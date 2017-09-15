import os, cv2, time, numpy, imutils, argparse
import logging as log
from multiprocessing import Pool
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score,classification_report,cohen_kappa_score,confusion_matrix
from scipy.cluster.vq import *

def get_args():
    # Set and get script arguments
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--testingSetPath", help="Path to testing Set")
    group.add_argument("-i", "--image", help="Path to image")
    parser.add_argument("-m", "--classifierModelFile", help="Classifier Model File", required="True")
    parser.add_argument("-r", "--results", help="Results File Name", required="True")
    parser.add_argument('-s',"--show", action='store_true')
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

    # Load the classifier, class names, scaler, number of clusters and vocabulary
    log.info("Loading classifier, class names, scaler, number of cluster and vocabulary")
    clf, classes_names, stdSlr, k, voc = joblib.load(args.classifierModelFile + ".pkl")
    # Get the path of the testing image(s) and store them in a list
    log.info("Loading images")
    image_paths = []
    if args.testingSetPath:
        try:
            testing_names = os.listdir(args.testingSetPath)
        except OSError:
            log.error("No such directory {}. Check if the directory exists".format(args.testingSetPath))
            exit()
        for testing_name in testing_names:
            directory_name = os.path.join(args.testingSetPath, testing_name)
            class_path = imutils.imlist(directory_name)
            image_paths+=class_path
    else:
        image_paths = [args.image]

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

    # Creating codebook
    log.info("Creating codebook")
    test_features = numpy.zeros((set_size, k), "float32")
    i = 0
    for feature in features:
        for _ , descriptor in feature:
            words, _ = vq(descriptor, voc)
            for w in words:
                test_features[i][w] +=1
            i += 1

    # Scaling the words
    log.info("Scaling words")
    test_features = stdSlr.transform(test_features)

    # Performing the predictions
    log.info("Performing predictions")
    predictions =  [classes_names[i] for i in clf.predict(test_features)]

    # Stopping counting execution time
    log.info("Stopping counting execution time")
    log.info("---Time: %s seconds ---" % (time.time() - start_time))

    # Getting only image class
    for i in range(set_size):
        print(image_paths[i])
        image_paths[i] = image_paths[i].split('/')[-2]
        print(image_paths[i])
        exit(1)

    # Calculating metrics
    log.info("Calculating metrics")
    accuracy = accuracy_score(image_paths,predictions)
    report = classification_report(image_paths,predictions)
    cohen_kappa = cohen_kappa_score(image_paths,predictions)
    cnf_matrix = confusion_matrix(image_paths,predictions)

    log.info("\nAccuracy: {}".format(accuracy))
    log.info("Cohen Kappa Score: {}\n".format(cohen_kappa))
    log.info("Report: {}".format(report))

    # Saving metrics
    log.info("Saving metrics")

    classes_labels = ""
    for label in classes_names:
        classes_labels += label + ";"
    numpy.savetxt(args.results + "_confusion_matrix.csv", cnf_matrix, delimiter=";", fmt="%10.f", header=classes_labels)

    accuracy_and_kappa = "Accuracy: {};Cohen Kappa Score: {}".format(accuracy,cohen_kappa)
    report_splitted = [report_line.split() for report_line in report.splitlines()]
    numpy.savetxt(args.results + "_report.csv", report_splitted, delimiter=';', fmt="%s", header=accuracy_and_kappa)

    # Show the results, if "show" flag set to true by the user
    if args.show:
        for image_path, prediction in zip(image_paths, predictions):
            image = cv2.imread(image_path)
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            pt = (0, 3 * image.shape[0] // 4)
            cv2.putText(image, prediction, pt ,cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, [0, 255, 0], 2)
            cv2.imshow("Image", image)
            cv2.waitKey(3000)
