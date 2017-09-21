import os, cv2, argparse, imutils, numpy
import logging as log
from multiprocessing import Pool
from sklearn.externals import joblib

def get_args():
    # Set and get script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--imageSetPath", help="Path to Image Set", required="True")
    parser.add_argument("-f", "--featuresPath", help="Path to Persist Features Extracted", required=True)
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    return parser.parse_args()

def stack_descriptors(features):
    # Stack all the descriptors vertically in a numpy array
    descriptors = features.pop(0)[1]
    for _ , descriptor in features:
        descriptors = numpy.concatenate((descriptors, descriptor), axis=0)
    return descriptors

def detect_and_compute_keypoints(image_paths):
    # Detect and compute keypoints found on images
    global keypoint_detector
    global keypoint_descriptor
    keypoints_descriptions = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        keypoints = keypoint_detector.detect(image, None)
        (_ , description) = keypoint_descriptor.compute(image, keypoints)
        #print("imagepath: {} - kps: {} - descriptors: {}".format(len(keypoints),image_path,description.shape))
        keypoints_descriptions.append((image_path, description))
    return keypoints_descriptions

if __name__ == "__main__":
    # Get arguments
    args = get_args()

    # If verbose argument is provided, allow logs to be printed
    if args.verbose:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    # Get the training classes names and store them in a list
    log.info("Getting training classes names and store them in a list")
    try:
        images_names = os.listdir(args.imageSetPath)
    except OSError:
        log.error("No such directory {}. Check if the directory exists".format(args.imageSetPath))
        exit()

    if not os.path.isdir(args.featuresPath):
        try:
            os.makedirs(args.featuresPath)
        except OSError:
            log.error("Can't create a folder on " + args.featuresPath)
            exit()

    images_paths = []
    images_classes = []
    class_id = 0

    for image_name in images_names:
        directory_name = os.path.join(args.imageSetPath, image_name)
        class_path = imutils.imlist(directory_name)
        images_paths += class_path
        images_classes += [class_id]*len(class_path)
        class_id += 1

    # Get the amount of cpus
    cpus = os.cpu_count()

    # Take the set size
    set_size = len(images_paths)

    # Calculates the number of subsets required for the quantity of cpus
    subset_size = int(numpy.ceil(set_size/cpus))

    # Divide the set into subsets according to the quantity of cpus
    log.info("Dividing feature detection and extraction between {} processes".format(cpus))
    images_paths_parts = [images_paths[i:i + subset_size] for i in range(0, set_size, subset_size)]

    # Descriptors declaration
    global keypoint_detector
    global keypoint_descriptor

    detector_number = 0
    while True:
        detector_name = "detector_"
        # Choosing the detector
        print("Detector")
        if detector_number == 0:
            keypoint_detector = cv2.xfeatures2d.SIFT_create()
            detector_name += "SIFT"
            print("SIFT")
        elif detector_number == 1:
            keypoint_detector = cv2.xfeatures2d.SURF_create()
            detector_name += "SURF"
            print("SURF")
        elif detector_number == 2:
            keypoint_detector = cv2.FastFeatureDetector_create()
            detector_name += "FAST"
            print("FAST")
        # elif detector_number == 3:
        #     keypoint_detector = cv2.ORB_create()
        #     detector_name += "ORB"
        #     print("ORB")

        print("Descriptor")
        descriptor_number = 0
        while True:
            descriptor_name = "+descriptor_"
            #Choosing the descriptor
            if descriptor_number == 0:
                keypoint_descriptor = cv2.xfeatures2d.SIFT_create()
                descriptor_name += "SIFT"
                print("SIFT")
            elif descriptor_number == 1:
                keypoint_descriptor = cv2.xfeatures2d.SURF_create()
                descriptor_name += "SURF"
                print("SURF")

            # Create feature extraction and keypoint detector objects
            log.info("Create feature extraction and keypoint detector objects")
            pool = Pool(processes=cpus)

            # Detecting points and extracting features
            log.info("Detecting points and extracting features")
            keypoints_descriptions = pool.map(detect_and_compute_keypoints, (images_paths_parts))

            # Stack all the descriptors vertically in a numpy array using Pool
            log.info("Stack all descriptors vertically in a numpy array")
            descriptors = pool.map(stack_descriptors, (keypoints_descriptions))
            descriptors_result = descriptors.pop(0)
            for descriptor in descriptors:
                descriptors_result = numpy.concatenate((descriptors_result, descriptor), axis=0)

            # Save the SVM
            log.info("Saving descriptions")
            features_path = args.featuresPath+detector_name+descriptor_name+".pkl"
            joblib.dump((descriptors_result),features_path, compress = 0)

            descriptor_number += 1
            if descriptor_number == 2:
                break

        detector_number += 1
        if detector_number == 3:
            break
