import sys
sys.path.append('src')

import os
import json
import tensorflow as tf
from models.pose_classification_model_architecture import PoseClassificationModel
from data.pose_classification_data_loader import data_loader

# Note the preprocessing was already done in jupyter notebooks, so we run this file after we have run the preprocessing jupyter notebook
# If you run this before runnning the 'data_preprocessing_1.ipynb' and 'data_preprocessing_2.ipynb' it will fail, this is very important

# Defining the dataset paths
dataset_path = './data/processed'

x_train_suffix = 'train_4_channel_info'
y_train_suffix = 'train_labels_pose_class'

x_val_suffix = 'val_4_channel_info'
y_val_suffix = 'val_labels_pose_class'

# Load the preprocessed data
x_train = os.listdir(f'{dataset_path}/{x_train_suffix}')
y_train = os.listdir(f'{dataset_path}/{y_train_suffix}')
x_train.sort(), y_train.sort()

x_val = os.listdir(f'{dataset_path}/{x_val_suffix}')
y_val = os.listdir(f'{dataset_path}/{y_val_suffix}')
x_val.sort(), y_val.sort()

# Use the dataloader
train_dataset, val_dataset = data_loader(dataset_path, x_train, y_train, x_val, y_val, x_train_suffix, y_train_suffix, x_val_suffix, y_val_suffix)

# Use the model
model = PoseClassificationModel()

# Printing the model summary
print(model.get_model_summary())

# Compile the model
model.compile_model()

# Train the model
count = len(os.listdir('./logs/pose_classification_stage_checkpoints'))

history = model.train_model(train_dataset, val_dataset, 25, './logs/pose_classsification_logs_tensorflow_hub', 'pose_classification_based_on_keypoint_estimation',
                            f'./logs/pose_classification_stage_checkpoints/checkpoint_pose_classification_based_on_keypoints_predictions_{count + 1}.ckpt.weights.h5')

# # Write the history file to the logs
history_dict = history.history
history_json = json.dumps(history_dict, indent=4)

count = len(os.listdir('./logs/pose_classsification_logs_tensorflow_hub'))
try:
    path = f'./logs/pose_classsification_logs_tensorflow_hub/results_{count + 1}.json'
    with open(path, 'w') as file:
        file.write(history_json)
    print(f"[+] Results of training saved successfully to {path}")
except Exception as e:
    print("[-] Failed to write the training results...")
    print(e)


# Note performance metrics on validation set were already calculated in model_building_2.ipynb so we're not calculating them here