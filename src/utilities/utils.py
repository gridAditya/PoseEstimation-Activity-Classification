import math
import cv2
from matplotlib import pyplot as plt
import numpy as np
import json, yaml
import os
import copy
from typing import Any, List, Tuple, Union

def write_data_if_file_not_exists(file_path, data):
  """
  This function checks if a file exists at the specified path.
  If it doesn't exist, it writes the provided data to the file in JSON format.

  Args:
      file_path (str): The path to the file.
      data (any): The data to be written to the file.
  """
  if not os.path.exists(file_path):
    with open(file_path, 'w') as f:
      json.dump(data, f)
      print(f"Data written to {file_path} as the file didn't exist.")
      return False
  else:
    print(f"File {file_path} already exists, so data not written.")
    return True

def open_file(file_path: str) -> Union[dict, list, None]:
    """
    Opens and reads the content of a JSON or YAML file.

    Parameters:
    file_path (str): The path to the file.

    Returns:
    Union[dict, list, None]: The content of the file parsed to a dictionary or a list,
                             or None if an error occurs.
    """
    try:
        with open(file_path, 'r') as file:
            if file_path.endswith('.json'):
                return json.load(file)
            elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                return yaml.safe_load(file)
            else:
                raise ValueError(f'Unsupported file format: {file_path}')
    except Exception as e:
        print(f'An error occurred: {e}')
        return None

def rescale_image_and_annotations(image, annotations, indx, target_size=(480, 640)):
  """
  Rescales an image and its annotations (bounding boxes and keypoints) to the target size.

  Args:
      image: Original image as a NumPy array.
      annotations: Dictionary containing bounding boxes and keypoints.
      target_size: Desired size (height, width) after rescaling.

  Returns:
      Rescaled image and updated annotations.
  """

  height, width = image.shape[:2]
  width_scale = target_size[1] / width
  height_scale = target_size[0] / height

  rescaled_image = cv2.resize(image, (target_size[1], target_size[0])) # (width, height)

  # Update the bounding boxes
  for i in range(len(annotations[indx]["personInfo"])):
    annotations[indx]["personInfo"][i]["x1"] = int(int(annotations[indx]["personInfo"][i]["x1"]) * width_scale)
    annotations[indx]["personInfo"][i]["y1"] = int(int(annotations[indx]["personInfo"][i]["y1"]) * height_scale)
    annotations[indx]["personInfo"][i]["x2"] = int(int(annotations[indx]["personInfo"][i]["x2"]) * width_scale)
    annotations[indx]["personInfo"][i]["y2"] = int(int(annotations[indx]["personInfo"][i]["y2"]) * height_scale)
  
  # Update keypoints
  for i in range(len(annotations[indx]["personInfo"])):
    for j in range(len(annotations[indx]["personInfo"][i]['x'])):
      annotations[indx]["personInfo"][i]['x'][j] = int(int(annotations[indx]["personInfo"][i]['x'][j]) * width_scale)
      annotations[indx]["personInfo"][i]['y'][j] = int(int(annotations[indx]["personInfo"][i]['y'][j]) * height_scale)

  print(f"[+] height: {height} width: {width} tHeight: {target_size[0]} twidth: {target_size[1]}")
  # print(f"[+] width_scale: {width_scale} height_scale: {height_scale}")
  return rescaled_image

def rescale_and_save_image(file_path, file_path_write, annotations, indx, target_size=(480, 640)):
  """
  Rescales an image from a file path, its annotations (bounding boxes and keypoints),
  and saves the rescaled image at the same file path (overwriting the original).

  Args:
      file_path: path from where to read the image file.
      file_path_write: path where you want to write the image to
      annotations: an array containing the annotations of the images
      target_size: Desired size (height, width) after rescaling.
  """
  try:
    image = cv2.imread(file_path)
    # Rescale image and annotations
    rescaled_image = rescale_image_and_annotations(
        image, annotations, indx, target_size)
    
    # Save the rescaled image at the same file path (overwriting)
    cv2.imwrite(file_path_write, rescaled_image)
    print(f"Annotations rescaled and image is saved to: {file_path_write}")
  except (IOError, FileNotFoundError) as e:
    print(f"Error reading image file: {e}")

def clear_files_in_directory(dir_path):
    """
    Function to clear all the files in a directory

    Args:
        dir_path: path to the directory whose files are to be cleared
    """
    entries = os.listdir(dir_path)

    for entry in entries:
        full_path = os.path.join(dir_path, entry)

        # Check if its a file or not
        if os.path.isfile(full_path):
            os.remove(full_path)
            print(f"Deleted file: {full_path}")
    print("Deletion completed (if there were files).")

def calc_bbox_coordinates(x, y, image_height=256, image_width=256):
    """
    Calculates the rough bounding box co-ordinates of the given object from its keypoints co-ordinates data

    Args:
        x: An array containing the x-co-ordinates of all the keypoints, must be un-normalised
        y: An array containing the y-co-ordinates of all the keypoints, must be un-normalised
        image_height: height resolution of the image(image.shape[0])
        image_width: width resolution of the image(image.shape[1])

    Returns:
        A numpy array of integers [x1, y1, x2, y2] containing the bounding box co-ordinates of the given
    """
    x1, y1 = np.inf, np.inf
    x2, y2 = -np.inf, -np.inf
    y_max_indx, y_min_indx = -1, -1
    for i in range(len(x)):
        if x[i] != 0 and x1 > x[i]: # find x-min = x1
            x1 = int(x[i])
        if x[i] != 0 and x2 < x[i]: # find x-max = x2
            x2 = int(x[i])

    for i in range(len(y)):
        if y[i] != 0 and y1 > y[i]: # find y-min = y1
            y1 = int(y[i])
            y_min_indx = i
        if y[i] != 0 and y2 < y[i]: # find y-max = y2
            y2 = int(y[i])
            y_max_indx = i

    # Handle the x co-ordinates
    if x1 > (image_width / 2):
        if (int(x1 - 0.10 * x1) > 0):
            x1 = int(x1 - 0.10 * x1)
        elif (int(x1 - 0.05 * x1) > 0):
            x1 = int(x1 - 0.05 * x1)
    else:
        if (int(x1 - 0.11 * x1) > 0):
            x1 = int(x1 - 0.11 * x1)
        elif (int(x1 - 0.10 * x1) > 0):
            x1 = int(x1 - 0.10 * x1)
        elif (int(x1 - 0.05 * x1) > 0):
            x1 = int(x1 - 0.05 * x1)

    if x2 > (image_width / 2):
        if (int(x2 + 0.05 * x2) < image_width):
            x2 = int(x2 + 0.05 * x2)
    else:
        if (int(x2 + 0.13 * x2) < image_width):
            x2 = int(x2 + 0.13 * x2)
        elif (int(x2 + 0.7 * x2) < image_width):
            x2 = int(x2 + 0.05 * x2)

    # Handle the y co-ordinates
    if y_min_indx == 2 or y_min_indx == 3 or y_min_indx == 6: # check if rhip, lhip or pelvis is at top
        if (int(y1 - 0.25 * y1) > 0):
            y1 = int(y1 - 0.25 * y1)
        elif (int(y1 - 0.20 * y1) > 0):
            y1 = int(y1 - 0.20 * y1)
        elif (int(y1 - 0.15 * y1) > 0):
            y1 = int(y1 - 0.15 * y1)
        elif (int(y1 - 0.10 * y1) > 0):
            y1 = int(y1 - 0.10 * y1)
        elif (int(y1 - 0.05 * y1) > 0):
            y1 = int(y1 - 0.05 * y1)
    else:
        if (int(y1 - 0.20 * y1) > 0):
            y1 = int(y1 - 0.20 * y1)
        elif (int(y1 - 0.15 * y1) > 0):
            y1 = int(y1 - 0.15 * y1)
        elif (int(y1 - 0.10 * y1) > 0):
            y1 = int(y1 - 0.10 * y1)
        elif (int(y1 - 0.05 * y1) > 0):
            y1 = int(y1 - 0.05 * y1)
    
    if y_max_indx == 2 or y_max_indx == 3 or y_max_indx == 6:
        if (int(y2 + 0.20 * y2) < image_height):
            y2 = int(y2 + 0.20 * y2)
        elif (int(y2 + 0.15 * y2) < image_height):
            y2 = int(y2 + 0.15 * y2)
        elif (int(y2 + 0.10 * y2) < image_height):
            y2 = int(y2 + 0.10 * y2)
        elif (int(y2 + 0.05 * y2) < image_height):
            y2 = int(y2 + 0.05 * y2)
    else:
        if (int(y2 + 0.15 * y2) < image_height):
            y2 = int(y2 + 0.15 * y2)
        elif (int(y2 + 0.10 * y2) < image_height):
            y2 = int(y2 + 0.10 * y2)
        elif (int(y2 + 0.05 * y2) < image_height):
            y2 = int(y2 + 0.05 * y2)
    # print(f"[+] x1: {x1} x2: {x2} y1: {y1} y2: {y2}")
    # print()
    return np.array([x1, y1, x2, y2])

def draw_bounding_box(image, x1, y1, x2, y2, color=(0, 0, 255), thickness=1):
  """
  Draws a rectangle on the given image representing the bounding box with specified coordinates.

  Args:
      image: The image on which to draw the rectangle (NumPy array).
      x1: Top-left x-coordinate of the bounding box.
      y1: Top-left y-coordinate of the bounding box.
      x2: Bottom-right x-coordinate of the bounding box.
      y2: Bottom-right y-coordinate of the bounding box.
      color: The color of the rectangle (BGR format, default: green).
      thickness: The thickness of the rectangle line (default: 2).
  """

  # Ensure coordinates are integers for OpenCV
  arr = (np.array([x1 ,y1 ,x2 , y2])).astype(int)
  x1, y1, x2, y2 = arr

  print(type(x1))
  # Draw the rectangle using cv2.rectangle()
  cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

def drawPointsAndNumbers(image, skeleton_data, locations, ids):
  """
  Draws a point and writes the corresponding ID at each location in the image.

  Args:
      image: The image to draw on.
      locations: A list of tuples (x, y) representing the locations of the points.
      ids: A list of integers representing the IDs to write at each location.
  """

  locations = np.array(locations)
  locations = np.rint(locations)
  locations = locations.astype(int)
  
  # Ensure the number of locations and IDs match
  if len(locations) != len(ids):
    raise ValueError("[-] Locations and ids lists must have the same length.")

  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 0.5
  font_thickness = 1

  for (x, y), id in zip(locations, ids):
    color = (0, 0, 255)
    radius = 4

    # Draw a circle at the location
    cv2.circle(image, (x, y), radius, color, cv2.FILLED)

    # Write the corresponding ID slightly above the point
    text = str(id) 
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x = x - text_size[0] // 2
    text_y = y + radius + text_size[1]
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
  
  # print("[+] Till here...")
  # Draw lines between joints to get a skeleton
  for i in range(len(skeleton_data)):
    jt_1_exist, jt_2_exist = False, False
    p1, p2 = (0, 0), (0, 0)
  
    for j in range(len(ids)):
      if (ids[j] == skeleton_data[i][0]):
        p1 = locations[j]
        jt_1_exist = True
      elif (ids[j] == skeleton_data[i][1]):
        p2 = locations[j]
        jt_2_exist = True
    
    if (jt_1_exist and jt_2_exist):
      color = (0, 255, 0)
      cv2.line(image, p1, p2, color, thickness=2)

def randPlotImage(valid_annot_data, skeleton_data, root, dir, rscale_x=1.0, rscale_y=1.0, image_height=256, image_width=256):
    """
        Randomly plots an image from the given dataset
    """
    choice = np.random.randint(low=0, high=len(valid_annot_data) - 1, size=1)[0]
    # choice = 1623
    image_path = f'{root}/{dir}/' + valid_annot_data[choice]['name']
    image = cv2.imread(image_path)
    num_person = len(valid_annot_data[choice]['personInfo'])

    print(f"[+] Choice: {choice}, num_person: {num_person} image:{valid_annot_data[choice]['name']} with shape {image.shape}")

    for i in range(num_person):
        xloc = np.array(valid_annot_data[choice]['personInfo'][i]['x']) * rscale_x # rescale back to original dimensions
        xloc = np.rint(xloc) # round off to nearest integer

        yloc = np.array(valid_annot_data[choice]['personInfo'][i]['y']) * rscale_y # rescale back to original dimensions
        yloc = np.rint(yloc) # round off to nearest integer
        
        visibility = valid_annot_data[choice]['personInfo'][i]['is_visible']
        locations = []
        visible_joint_id = []
        for j in range(len(xloc)):
            if xloc[j] != 0 and yloc[j] != 0: # Mark the valid joints
                locations.append((xloc[j], yloc[j]))
                visible_joint_id.append(valid_annot_data[choice]['personInfo'][i]['id'][j])
        drawPointsAndNumbers(image, skeleton_data, locations, visible_joint_id)

        bbox_arr = calc_bbox_coordinates(xloc, yloc, image_height, image_width)
        x1, x2 = bbox_arr[0], bbox_arr[2] 
        y1, y2 = bbox_arr[1], bbox_arr[3]
        draw_bounding_box(image, x1, y1, x2, y2)
        
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Image Name: {valid_annot_data[choice]['name']}")
    return valid_annot_data[choice]['name']

def return_data_in_yolov8_format(valid_annot_data, indx, rscale_x=1.0, rscale_y=1.0):
    """
    Takes in an image_indx and returns the string information to be stored in a .txt file
    in yolov8 format

    Args:
        valid_annot_data: annotations data for the entire dataset
        indx: indx where the image annotations is stored in the valid_annot_data array
        rscale_x: scaling factor by which you have normalised the images's x-dimension
        rscale_y: scaling factor by which you have normalised the images's y-dimension
    Returns:
        res: an array containing annotations data corresponding to each person in the image in strings format
    """

    annot_data = copy.deepcopy(valid_annot_data[indx]) # referencing can cause unexpected changes...
    res = []

    for i in range(len(annot_data["personInfo"])):
        data = "0"
        x_rescaled = copy.deepcopy(np.array(annot_data["personInfo"][i]['x'])) # referencing can cause unexpected changes...
        x_rescaled = (x_rescaled * rscale_x).astype(int)

        y_rescaled = copy.deepcopy(np.array(annot_data["personInfo"][i]['y'])) # referencing can cause unexpected changes...
        y_rescaled = (y_rescaled * rscale_y).astype(int)

        bbox_arr = calc_bbox_coordinates(x_rescaled, y_rescaled)
        x_center = (bbox_arr[0] + bbox_arr[2]) / 2
        y_center = (bbox_arr[1] + bbox_arr[3]) / 2
        width = abs(bbox_arr[0] - bbox_arr[2])
        height = abs(bbox_arr[1] - bbox_arr[3])
        
        
        # Normalise the bbox_co-ordinates
        x_center = (1.0 * x_center) / rscale_x
        y_center = (1.0 * y_center) / rscale_y
        width = (1.0 * width) / rscale_x
        height = (1.0 * height) / rscale_y

        data = data + f" {x_center} {y_center} {width} {height}"

        # Now we append the normalise x and y normalised keypoints and visibilty
        for j in range(len(annot_data["personInfo"][i]['x'])):
            # print(annot_data["personInfo"][i]['x'][j])
            keypoint_x = str(annot_data["personInfo"][i]['x'][j])
            if keypoint_x == 0:
                keypoint_x = 0
            
            keypoint_y = str(annot_data["personInfo"][i]['y'][j])
            if keypoint_y == 0:
                keypoint_y = 0
                
            is_visible = str(annot_data["personInfo"][i]['is_visible'][j])
            data = data + f" {keypoint_x} {keypoint_y} {is_visible}"
        res.append(data)
    return res

def calcEuclideanDistance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

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

def plot_bar_graph(data_dict, name):
    plt.figure(figsize=(10, 8))

    activities = list(data_dict.keys())
    values = list(data_dict.values())

    colors = plt.cm.tab20c(range(len(data_dict)))
    plt.bar(activities, values, color=colors)
    plt.xlabel('Activities')
    plt.ylabel('Values')
    plt.title(f'Values of Activities in {name}')
    plt.xticks(rotation=90)
    plt.tight_layout()

    plt.show()

def random_plot_image_from_npy_files(sample_dir_path, label_dir_path, class_names, valid_annot_data):
    sample_entries = os.listdir(sample_dir_path)
    labels_entries = os.listdir(label_dir_path)

    choice = np.random.randint(low=0, high=len(sample_entries) - 1, size=1)[0]
    label_indx = labels_entries.index(sample_entries[choice][:-4] + '.txt') # change here------------------
    file_path = f"{label_dir_path}/{labels_entries[label_indx]}"

    # Read the label
    label = None
    try:
        with open(file_path, 'r') as file:
            label = file.read()
    except Exception as e:
        print(f"[-] Failed to read content of {labels_entries[choice]} aborting...")
        return

    # Read the sample
    data = np.load(f"{sample_dir_path}/{sample_entries[choice]}") # float32 numpy array # change here---------------
    data = data * 255.0
    data = data.astype(np.uint8)

    context_image = data[:, :, :3]
    poseHeatImage = data[:, :, 3]

    fig, axarr = plt.subplots(1, 2)
    axarr[0].imshow(context_image)
    axarr[1].imshow(poseHeatImage)

    # Find the activity name
    label_title = ''
    for activity in class_names.keys():
        if class_names[activity] == int(label):
            label_title = str(activity)
            break
    
    # Find the actual label
    indx = -1
    for i in range(len(valid_annot_data)):
        name = sample_entries[choice][:-4] + '.jpg' # change here------------
        if valid_annot_data[i]['name'] == name:
            indx = i
            break
    
    print(f"[+] Choice {choice} loading {sample_entries[choice]} label:{label}")
    print("[+] Saved label: ", label_title, " and actual_label: ", valid_annot_data[indx]['general_activity_name'])
    fig.suptitle(label_title)
    plt.tight_layout()
    plt.show()

