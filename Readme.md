# Pose Estimation and Activity Classification

The models developed in this project aim to estimate the pose(keypoint co-ordinates) as well as classify the kind of activity the humans in the give image are doing.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install.

```bash
pip setup.py sdist
pip install /path/to/dist/myproject-1.0.tar.gz
```

## Purpose
```
Uses the input image to estimate the pose of the human body.
Combines the images with the pose to estimate the kind of activity the humans in the image are doing.
```




## Total Classes
```
class_names = {
 'sports': 0,
 'miscellaneous': 1,
 'home activities': 2,
 'occupation': 3,
 'fishing and hunting': 4,
 'home repair': 5,
 'conditioning exercise': 6,
 'lawn and garden': 7,
 'religious activities': 8,
 'music playing': 9,
 'inactivity quiet/light': 10,
 'water activities': 11,
 'running': 12,
 'winter activities': 13,
 'walking': 14,
 'dancing': 15,
 'bicycling': 16,
 'transportation': 17,
 'self care': 18,
 'volunteer activities': 19
}
