# Bag of Features

Python Implementation of Bag of Words for Image Recongnition using OpenCV and sklearn

## Original author
This is a clean fork of [bikz05's](https://github.com/bikz05/bag-of-words) work.

## Installation

Dependencies:
* numpy
* sklearn
* scipy

### Note
I am using OpenCV3 and a number of things have been moved to the [opencv_contrib](https://github.com/Itseez/opencv_contrib/) repo.
Make sure you install the _xfeatures2d_ module to be able to use [SIFT](http://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html#gsc.tab=0).

## Training the Classifier
```
python learn.py -t dataset/train
```
## Testing the Classifier
* Testing a single image
```
python predict.py -t dataset/test/class/image.extension --visualize
```

* Testing a testing dataset
```
python predict.py -t dataset/test --visualize
```
The `--visualize` flag will display the image with the corresponding label printed on the image/

# Troubleshooting

If you get

```python
AttributeError: 'LinearSVC' object has no attribute 'classes_'
```

error, the simply retrain the model.
