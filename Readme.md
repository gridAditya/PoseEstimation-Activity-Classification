# Pose Estimation and Activity Classification

The models developed in this project aim to estimate the pose(keypoint co-ordinates) as well as classify the kind of activity the humans in the given image are doing.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install.

```bash
pip install -r requirements.txt
```

## Purpose
```
Uses the input image to estimate the pose of the human body.
Combines the images with the pose to estimate the kind of activity the humans in the image
are doing.
```

## How to structure the downloaded data
````
After downloading the MPII Human Pose Dataset, it will have two folders 'mpii_human_pose_v1_u12_2' and 'images'.
Rename the 'images' folder to 'original_images' and then place both the folders inside '/data/raw'.

Structure of the processed data directory:
root/data/processed
       |________________ images
       |                    |_______train
       |                    |_______val
       |________________ labels
       |                    |_______train
       |                    |_______val
       |________________ images_pose
       |                     |_______train
       |                     |_______val
       |________________ labels_pose
       |                     |_______train
       |                     |_______val
       |________________ train_4_channel_info
       |
       |________________ train_labels_pose_class
       |
       |________________ val_4_channel_info
       |
       |________________ val_labels_pose_class
````

## How to use the project
```
You can use the 'model_inference_final.ipynb' jupyter notebook to make inferences on the any image, just change
the value in the 'image_path' variable, or you can use the trained models
(last.pt and pose_classification_13:14:15_18831.keras) available in the root/models directory.
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
