import argparse as ap
import cv2
import imutils
import numpy as np
import os, time
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *

start_time = time.time()

# Load the classifier, class names, scaler, number of clusters and vocabulary
clf, classes_names, stdSlr, k, voc = joblib.load("bof.pkl")

# Get the path of the testing set
parser = ap.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-t", "--testingSet", help="Path to testing Set")
group.add_argument("-i", "--image", help="Path to image")
parser.add_argument('-v',"--visualize", action='store_true')
args = vars(parser.parse_args())

# Get the path of the testing image(s) and store them in a list
print("Carregando imagens")
image_paths = []
if args["testingSet"]:
    test_path = args["testingSet"]
    try:
        testing_names = os.listdir(test_path)
    except OSError:
        print ("No such directory \(test_path)\nCheck if the file exists")
        exit()
    for testing_name in testing_names:
        dir = os.path.join(test_path, testing_name)
        class_path = imutils.imlist(dir)
        image_paths+=class_path
else:
    image_paths = [args["image"]]

#raise SystemExit(0)
# Create feature extraction and keypoint detector objects
#fea_det = cv2.FeatureDetector_create("SIFT")
#des_ext = cv2.DescriptorExtractor_create("SIFT")
print("Extraindo caracteristicas")
sift = cv2.xfeatures2d.SIFT_create()

# List where all the descriptors are stored
des_list = []

for image_path in image_paths:
    im = cv2.imread(image_path)
    if im is None:
        print ("No such directory \(test_path)\nCheck if the file exists")
        exit()
    #kpts = fea_det.detect(im)
    #kpts, des = des_ext.compute(im, kpts)
    (kpts, des) = sift.detectAndCompute(im, None)
    des_list.append((image_path, des))

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[0:]:
    descriptors = np.vstack((descriptors, descriptor))

print("Criando codebook")
test_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        test_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scale the features
test_features = stdSlr.transform(test_features)

# Perform the predictions
print("Realizando predições")
predictions =  [classes_names[i] for i in clf.predict(test_features)]

# Visualize the results, if "visualize" flag set to true by the user
if args["visualize"]:
    for image_path, prediction in zip(image_paths, predictions):
        image = cv2.imread(image_path)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        pt = (0, 3 * image.shape[0] // 4)
        cv2.putText(image, prediction, pt ,cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, [0, 255, 0], 2)
        cv2.imshow("Image", image)
        cv2.waitKey(3000)
else:
    print("Calculando Acurácia...")
    total = 0
    hits = 0
    errors = 0
    for image_path, prediction in zip(image_paths, predictions):
        classe = image_path.split('/')[3]
        total += 1
        if prediction == classe:
            hits += 1
        else:
            errors += 1
    avg = (hits / total) * 100
    print("Total: {} - Acertos: {} - Erros: {} - Acuracia: {}".format(str(total),str(hits),str(errors), str(avg)))

print("--- %s seconds ---" % (time.time() - start_time))
