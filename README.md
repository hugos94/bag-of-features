# bagOfFeatures

Python Implementation of Bag of Features for Image Recongnition using OpenCV and sklearn

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
