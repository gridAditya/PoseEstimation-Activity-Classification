import numpy as np
import cv2, base64
import math
from fastapi import HTTPException
from matplotlib import pyplot as plt

def rescale_image(image, target_size=(480, 640)):
  rescaled_image = cv2.resize(image, (target_size[1], target_size[0])) # (width, height)
  return rescaled_image

def draw_skeleton(skeleton_data, keypoints, image_height, image_width):
    poseHeatImage = np.zeros((image_height, image_width))
    
    # Drawing the skeleton
    for (jt1, jt2) in skeleton_data:
        for i in range(len(keypoints)):
            x1, y1 = int(keypoints[i][jt1][0]), int(keypoints[i][jt1][1])
            x2, y2 = int(keypoints[i][jt2][0]), int(keypoints[i][jt2][1])

            if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0): # Skip the joints whose keypoints are not predicted
                continue
            poseHeatImage = cv2.line(poseHeatImage, (x1, y1), (x2, y2), color=190, thickness=2) 

    # Placing the joints
    for i in range(len(keypoints)):
        for j in range(len(keypoints[i])):
            x, y = int(keypoints[i][j][0]), int(keypoints[i][j][1])
            poseHeatImage = cv2.circle(poseHeatImage, (x, y), radius=2, color=250, thickness=2)
    return poseHeatImage

def run_inference(skeleton_data, keypoints_model, pose_classification_model, image, class_names, image_height, image_width):
    # Apply a gaussian blur
    rescaled_image = rescale_image(image, (image_height, image_width))

    # Estimate the keypoint locations using stage-1
    results = keypoints_model(rescaled_image)
    num_humans = results[0].keypoints.xy.shape[0]
    print(f"[+] Human_Count: {num_humans}")

    if results[0].keypoints.xy.shape[1] == 0:
        print(f"[-] Nothing detected, aborting....")
        raise HTTPException(status_code=401, detail={"error": "Image does not contain any pose."})
    
    poseHeatImage = draw_skeleton(skeleton_data, results[0].keypoints.xy, image_height, image_width)
    poseHeatImage = poseHeatImage.reshape(image_height, image_width, 1)

    pose_data = np.concatenate([rescaled_image, poseHeatImage], axis=-1)
    pose_data = pose_data.astype(np.float32)
    pose_data = pose_data / 255.0
    channels = 4

    # Data to feed to the classification stage
    data_to_feed = pose_data.reshape(1, image_height, image_width, channels) # (1, 480, 640, 4)

    # Make the label predictions
    predictions = pose_classification_model.predict(data_to_feed)
    predictions = predictions[0]
    
    label = np.argmax(predictions)
    label_title = None

    for activity in class_names.keys():
        if class_names[activity] == label:
            label_title = activity
            break
    return [rescaled_image, poseHeatImage, label_title]

def check_if_valid_image(class_present, confidence_scores):
    class_present = np.array(class_present)
    confidence_scores = np.array(confidence_scores)

    for i in range(len(class_present)):
        if (class_present[i] == 0 and confidence_scores[i] >= 0.4):
            return True
    return False

