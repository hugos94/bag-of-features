import os, argparse
import logging as log
from sklearn.externals import joblib

def get_args():
    # Set and get script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--featuresPath", help="Path to Persist Features Extracted", required=True)
    parser.add_argument("-c", "--classifierModelFile", help="Classifier Model File", required="True")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    return parser.parse_args()

if __name__ == '__main__':
    # Get arguments
    args = get_args()

    # If verbose argument is provided, allow logs to be printed
    if args.verbose:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    # Get the training classes names and store them in a list
    log.info("Getting features names and store them in a list")
    try:
        features_names = os.listdir(args.featuresPath)
    except OSError:
        log.error("No such directory {}. Check if the directory exists".format(args.featuresPath))
        exit()

    #descriptors_result = joblib.load(args.featuresPath)
