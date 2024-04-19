# Pose Estimation and Activity Classification

The models developed in this project aim to estimate the pose(keypoint co-ordinates) as well as classify the kind of activity the humans in the given image are doing.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install.

```bash
pip install -r requirements.txt
```
For running the API in "Project2_V_1.7_Api", since we have implemented our own custom frontend and database with MongoDb, we have to install the MongoDb and run it, so follow the below steps:
```bash
1: First install homeBrew:
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

2: Then run the "install_mongodb.sh" bash script to install mongodb database.
3: After this run the "start_mongodb.sh" bash script to run the mongodb service.
4: After examining the api you can close the mongodb service using "stop_mongodb.sh" bash script.
```

## Purpose
```
Uses the input image to estimate the pose of the human body.
Combines the images with the pose to estimate the kind of activity the humans in the image
are doing.
```
## Changes required to config.yaml
```
When using the YOLO.train function and passing a YAML file to the data parameter, there is an issue.
It is not possible to pass a path that is relative to the working directory, as this causes errors when the paths in the YAML file are read.

To fix this issue you must change the "path" parameter in the config.yaml file in "/notebooks/config.yaml" and in "/src/models/config.yaml" to the absolute path
of the "/data/processed" directory in your system, otherwise you won't be able to train the yolov8-nano model.
```
## How to structure the downloaded data
````
After downloading the MPII Human Pose Dataset, it will have two folders 'mpii_human_pose_v1_u12_2' and 'images'.
Rename the 'images' folder to 'original_images' and then place both the folders inside '/data/raw'.

Structure of the processed data directory(you need to create folders in the structure given below in the processed data directory, they can be empty):
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
Download Links:   
[Images](https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz).   
[Annotations](https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip).

## How to use the project
```
You can use the 'model_inference_final.ipynb' jupyter notebook to make inferences on the any image, just change
the value in the 'image_path' variable, or you can use the trained models
(last.pt and pose_classification_13:14:15_18831.keras) available in the root/models directory.

Note: The image must be present locally on the system, as this does not support image-urls.
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
