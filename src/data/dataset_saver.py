import sys
sys.path.append('src')

import tensorflow as tf
import numpy as np
import cv2


from utilities.utils import calc_bbox_coordinates
from utilities.utils import calcEuclideanDistance
from utilities.utils import draw_skeleton

def save_dataset(dataset_path, dataset, model, class_names, image_name_to_activity_hash, sample_dir_suffix, label_dir_suffix, image_path_to_feed,
                 skeleton_data, image_height, image_width):
    skipped_names = []
    for image_name in dataset:
        path = f"{image_path_to_feed}/{image_name}"
        results = model(path)
        results = results[0]
        keypoint = results.keypoints.xy
        
        if (keypoint.shape[1] == 0): # No prediction can be made
            skipped_names.append(image_name)
            continue
        
        # Find the keypoints of the centermost person
        keypoint = tf.cast(tf.constant(keypoint), tf.int32)
        center_person_indx = -1
        center_person_x, center_person_y = 0, 0
        min_distance = np.inf

        for i in range(len(keypoint)):
            x = (keypoint[i][:, 0]).numpy().tolist()
            y = (keypoint[i][:, 1]).numpy().tolist()
            bbox_arr = calc_bbox_coordinates(x, y, image_height, image_width)
            center_x = (bbox_arr[0] + bbox_arr[2]) / 2
            center_y = (bbox_arr[1] + bbox_arr[3]) / 2
            distance = calcEuclideanDistance(center_x, center_y, image_width / 2, image_height / 2)
        
            if min_distance > distance:
                min_distance = distance
                center_person_indx = i
                center_person_x = center_x
                center_person_y = center_y
        
        # Now find the person with max_distance than the center-one(>=50px)
        indx_person_max_distance = -1
        max_distance = -np.inf
        for i in range(len(keypoint)):
            x = (keypoint[i][:, 0]).numpy().tolist()
            y = (keypoint[i][:, 1]).numpy().tolist()
            bbox_arr = calc_bbox_coordinates(x, y, image_height, image_width)
            center_x = (bbox_arr[0] + bbox_arr[2]) / 2
            center_y = (bbox_arr[1] + bbox_arr[3]) / 2
            distance = calcEuclideanDistance(center_x, center_y, center_person_x, center_person_y)

            if distance >= 50 and distance > max_distance:
                max_distance = distance
                indx_person_max_distance = i

        
        # Create a combined tensor
        if indx_person_max_distance == -1:
            keypoint = tf.reshape(keypoint[center_person_indx], [1, 16, 2]) # 16 joints 2 co-ordinates
        else:
            keypoint1 = tf.reshape(keypoint[center_person_indx], [1, 16, 2]) # 16 joints 2 co-ordinates
            keypoint2 = tf.reshape(keypoint[indx_person_max_distance], [1, 16, 2]) # 16 joints 2 co-ordinates
            keypoint = tf.concat([keypoint1, keypoint2], axis=0)
        
        # PoseHeat Information and image information
        poseHeatImage = (draw_skeleton(skeleton_data, keypoint, image_height, image_width)).reshape(image_height, image_width, 1) # np.ndarray
        poseHeatImage = poseHeatImage.astype(np.float32)

        image = cv2.imread(path) #np.ndarray
        image = image.astype(np.float32)

        # Create the information(sample, label) to save
        channel_4_info = np.concatenate([image, poseHeatImage], axis=-1) # creating a 4 channel information
        label = str(class_names[image_name_to_activity_hash[image_name]])

        # Normalise the channel_4_info before saving the data
        channel_4_info = channel_4_info / 255.0
        
        # Save the information and its label
        try:
            save_path_sample = f"{dataset_path}/{sample_dir_suffix}/{image_name[:-4]}.npy" # change here------------
            save_path_label = f"{dataset_path}/{label_dir_suffix}/{image_name[:-4]}.txt"
            
            # Saving the sample(.npy format, faster storage and access)
            np.save(save_path_sample, channel_4_info)

            # Saving the label
            try:
                with open(save_path_label, 'w') as file:
                    file.write(label)
            except Exception as e:
                print(f"Failed to save label: {e}")
                return -1
            print(f"[+] Successfully save sample {image_name} along with label: {label}")
        except Exception as e:
            print("Failed to save the sample", e)
            return -1
    return skipped_names

