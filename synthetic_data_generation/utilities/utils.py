import numpy as np
import cv2, copy, os, json
import datetime, random
import torch

from matplotlib import pyplot as plt

# Utilities functions
def read_json_file(path):
    data = None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        print("[+] Annotations data loaded succesfully!")
        return data
    except Exception as e:
        print(e)
        print("[-] Unable to loaded data. Exiting!")

def calc_bbox_coordinates(x, y, image_height=480, image_width=640):
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
      cv2.line(image, p1, p2, color, thickness=3)

def draw_bounding_box(image, x1, y1, x2, y2, color=(0, 0, 255), thickness=2):
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

def plot_image(image_dir_path, annotations_path, skeleton_data, category, rscale_y=1.0, rscale_x=1.0, image_height=480, image_width=640):
    images_list = os.listdir(image_dir_path)

    # Remove .DS_Store
    indx = images_list.index('.DS_Store')
    images_list.pop(indx)
    
    valid_annot_data = None

    # Read the json data
    valid_annot_data = read_json_file(annotations_path)

    # Make a random choice to choose an image
    choice = np.random.randint(low=0, high=len(images_list), size=1)[0]
    
    annot_data = valid_annot_data[choice]
    
    print(f"[+] Choice: {choice}, total images: {len(valid_annot_data)}")
    # Define the image path and load the image
    
    print(annot_data['name'])
    image_path = f"{image_dir_path}/{annot_data['name']}"
    image = cv2.imread(image_path)
    num_person = len(annot_data['personInfo'])

    print(f"[+] Num_person: {num_person} image:{annot_data['name']} with shape {image.shape}")

    for i in range(num_person):
        xloc = np.array(annot_data['personInfo'][i]['x']) * rscale_x # rescale back to original dimensions
        xloc = np.rint(xloc) # round off to nearest integer

        yloc = np.array(annot_data['personInfo'][i]['y']) * rscale_y # rescale back to original dimensions
        yloc = np.rint(yloc) # round off to nearest integer
        
        visibility = annot_data['personInfo'][i]['is_visible']
        locations = []
        visible_joint_id = []
        for j in range(len(xloc)):
            if xloc[j] != 0 and yloc[j] != 0: # Mark the valid joints
                locations.append((xloc[j], yloc[j]))
                visible_joint_id.append(annot_data['personInfo'][i]['id'][j])
        drawPointsAndNumbers(image, skeleton_data, locations, visible_joint_id)

        bbox_arr = calc_bbox_coordinates(xloc, yloc, image_height, image_width)
        x1, x2 = bbox_arr[0], bbox_arr[2] 
        y1, y2 = bbox_arr[1], bbox_arr[3]
        draw_bounding_box(image, x1, y1, x2, y2)
        
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Image Name: {annot_data['name']}")
    return annot_data['name']

# Define a function to return annot data given image name
def img_name_to_annot_data(image_name, valid_annot_data):
    annot_data = None
    index = -1
    for i in range(len(valid_annot_data)):
        if valid_annot_data[i]['name'] == image_name:
            annot_data = valid_annot_data[i]
            index = i
            break
    return annot_data, index

# Let's define a function to create a boolean mask out of image
def create_boolean_mask(mask):
    mask = mask[:, :, 0]
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j] != 0:
                mask[i][j] = 1
    return mask

def plot_image_with_name(image_path, annot_data, skeleton_data, category, rscale_y=1.0, rscale_x=1.0, image_height=480, image_width=640):
    # Define the image path and load the image
    
    print(annot_data['name'])
    
    image = cv2.imread(image_path)
    num_person = len(annot_data['personInfo'])

    print(f"[+] Num_person: {num_person} image:{annot_data['name']} with shape {image.shape}")

    for i in range(num_person):
        xloc = np.array(annot_data['personInfo'][i]['x']) * rscale_x # rescale back to original dimensions
        xloc = np.rint(xloc) # round off to nearest integer

        yloc = np.array(annot_data['personInfo'][i]['y']) * rscale_y # rescale back to original dimensions
        yloc = np.rint(yloc) # round off to nearest integer
        
        visibility = annot_data['personInfo'][i]['is_visible']
        locations = []
        visible_joint_id = []
        for j in range(len(xloc)):
            if xloc[j] != 0 and yloc[j] != 0: # Mark the valid joints
                locations.append((xloc[j], yloc[j]))
                visible_joint_id.append(annot_data['personInfo'][i]['id'][j])
        drawPointsAndNumbers(image, skeleton_data, locations, visible_joint_id)

        bbox_arr = calc_bbox_coordinates(xloc, yloc, image_height, image_width)
        x1, x2 = bbox_arr[0], bbox_arr[2] 
        y1, y2 = bbox_arr[1], bbox_arr[3]
        draw_bounding_box(image, x1, y1, x2, y2)
        
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Image Name: {annot_data['name']}, category: {category}")
    return annot_data['name']

# Synthetic image and annotation generation functions
def add_bg_to_image(image, mask, composite_mask, bg, x, y):
    mask = np.stack([mask, mask, mask], axis=-1)
    composite_image = copy.deepcopy(bg)
    
    h, w = image.shape[0], image.shape[1]
    x1, y1 = int(x - (w / 2)), int(y - (h / 2))
    # print(f"[+] Top_Left=> x1: {x1} y1: {y1}")
    if (x1 >= 0 and y1 >= 0):
        composite_image[y1: y1 + h, x1: x1 + w] = composite_image[y1: y1 + h, x1: x1 + w] * (mask ^ 1) + (image * mask)
        composite_mask[y1: y1 + h, x1: x1 + w] = composite_mask[y1: y1 + h, x1: x1 + w] * (mask ^ 1)
        return composite_image, composite_mask, composite_image
    
    if (x1 < 0 and y1 > 0):
        composite_image[y1: y1 + h, 0: x1 + w] = composite_image[y1: y1 + h, 0: x1 + w] * ((mask ^ 1)[:, abs(x1):]) + (image * mask)[:, abs(x1):]
        composite_mask[y1: y1 + h, 0: x1 + w] = composite_mask[y1: y1 + h, 0: x1 + w] * ((mask ^ 1)[:, abs(x1):])
        return composite_image, composite_mask, composite_image
    
    if (x1 > 0 and y1 < 0):
        composite_image[0: y1 + h, x1: x1 + w] = composite_image[0: y1 + h, x1: x1 + w] * ((mask ^ 1)[abs(y1):, :]) + (image * mask)[abs(y1):, :]
        composite_mask[0: y1 + h, x1: x1 + w] = composite_mask[0: y1 + h, x1: x1 + w] * ((mask ^ 1)[abs(y1):, :])
        return composite_image, composite_mask, composite_image
    return False

def annotate_composite_image(skeleton_data, composite_image, image, keypoint_x, keypoint_y, joint_id, x, y, image_height, image_width):
    x1, y1, x2, y2 = calc_bbox_coordinates(keypoint_x, keypoint_y, image_height, image_width)
    h, w = image.shape[0], image.shape[1]

    x_topleft_new, y_topleft_new = int(x - (w / 2)), int(y - (h / 2))
    keypoint_x, keypoint_y = np.array(keypoint_x), np.array(keypoint_y)
    x_trans, y_trans = x_topleft_new, y_topleft_new
    
    # print(f"[+] Original x1: {x1} y1: {y1}, x2: {x2} y2: {y2}")
    # Now apply the translation equation
    x1, x2 = x1 + x_trans, x2 + x_trans
    y1, y2 = y1 + y_trans, y2 + y_trans
    
    for i in range(len(keypoint_x)):
        if keypoint_x[i] != 0:
            keypoint_x[i] = keypoint_x[i] + x_trans
        if keypoint_y[i] != 0:
            keypoint_y[i] = keypoint_y[i] + y_trans
    # print(f"[+] Shifted x1: {x1} y1: {y1}, x2: {x2} y2: {y2}")
    locations = []
    visible_joint_id = []
    for i in range(len(keypoint_x)):
        if keypoint_x[i] != 0 and keypoint_y[i] != 0: # Mark the valid joints
            locations.append((keypoint_x[i], keypoint_y[i]))
            visible_joint_id.append(joint_id[i])
    drawPointsAndNumbers(composite_image, skeleton_data, locations, visible_joint_id)
    draw_bounding_box(composite_image, x1, y1, x2, y2)
    
    return composite_image

def return_equivalent_keypoints(annot_data, keypoint_x, keypoint_y, x, y, image_height=480, image_width=640):
    x1, y1, x2, y2 = calc_bbox_coordinates(keypoint_x, keypoint_y, image_height, image_width)
    x_topleft_new, y_topleft_new = int(x - (image_width / 2)), int(y - (image_height / 2))
    keypoint_x, keypoint_y = np.array(keypoint_x), np.array(keypoint_y)
    x_trans, y_trans = x_topleft_new, y_topleft_new

    # Now apply the translation equation
    x1, x2 = x1 + x_trans, x2 + x_trans
    y1, y2 = y1 + y_trans, y2 + y_trans

    
    for i in range(len(keypoint_x)):
        if keypoint_x[i] != 0:
            keypoint_x[i] = int(keypoint_x[i] + x_trans)
        if keypoint_y[i] != 0:
            keypoint_y[i] = int(keypoint_y[i] + y_trans)
        
        # Replace the invalid keypoints with 0
        if keypoint_x[i] <= 0:
            keypoint_x[i] = 0
        if keypoint_y[i] <= 0:
            keypoint_y[i] = 0

    # Modify the annotations data
    new_annot_data = dict()
    new_annot_data['x1'] = int(x1)
    new_annot_data['x2'] = int(x2)
    new_annot_data['y1'] = int(y1)
    new_annot_data['y2'] = int(y2)
    new_annot_data['x'] = copy.deepcopy(keypoint_x.tolist())
    new_annot_data['y'] = copy.deepcopy(keypoint_y.tolist())
    new_annot_data['id'] = copy.deepcopy(annot_data['id'])
    new_annot_data['is_visible'] = copy.deepcopy(annot_data['is_visible'])
    return new_annot_data

def generate_background_image(pipe, category, image_name, prompt_dict, num_images=1):
    if num_images > 5:
        print("[+] Generation above 5 image not supported due to time constraints!")

    print("[+] Image Name: ", image_name)
    seed = random.randint(0, 1000)
    generator = torch.Generator(device='mps').manual_seed(seed)
    # Generate the background image
    background_images = pipe(
                            prompt_dict[category][image_name],
                            num_inference_steps=50,
                            height=640,
                            width=640,
                            generator=generator,
                            num_images_per_prompt=num_images
                        ).images
    return background_images

def generate_synthetic_keypoints(unique_composite_image_name, annot_data, synthetic_annotations, bg_x_loc, bg_y_loc, image_height=480, image_width=640):
    # Now let's calculate the synthetic_annotations data
    synthetic_annotations.append({
                'name': unique_composite_image_name,
                'general_activity_name': annot_data['general_activity_name'],
                'personInfo': []
            })
    size = len(synthetic_annotations)

    for i in range(len(annot_data["personInfo"])):
        (synthetic_annotations[size - 1]["personInfo"]).append(return_equivalent_keypoints(annot_data["personInfo"][i], annot_data["personInfo"][i]['x'], annot_data["personInfo"][i]['y'], bg_x_loc, bg_y_loc, image_height, image_width))
    size = len(synthetic_annotations)
    return copy.deepcopy(synthetic_annotations[size - 1])

def find_image_annotations(valid_annot_data, image_name):
    for i in range(len(valid_annot_data)):
        if valid_annot_data[i]['name'] == image_name:
            return copy.deepcopy(valid_annot_data[i])
    return {}

def save_image(image, path):
    cv2.imwrite(path, image)
    print("[+] Synthetic Image saved successfully!")

def save_annotations(annotations, path):
    ## Append the annotations to an existing
    existing_data = None
    
    # Check if the file already exists
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                existing_data = json.load(f)
            print("[+] Existing annotations data loaded successfully!")

            # Append new data to existing one
            existing_data.append(annotations)
            print("[+] New annotations appended successfully!")

            # Save the modified data
            with open(path, 'w') as f:
                json.dump(existing_data, f, indent=4)
            print("[+] Modified data saved successfully!")
        except Exception as e:
            print(e)
            print("[-] Unable to save the modified annotations!")
    else:
        print("[+] Specified json file does not exists creating a new one!")
        try:
            with open(path, 'w') as f:
                json.dump([annotations], f, indent=4)
            print("[+] New data saved successfully")
        except Exception as e:
            print(e)
            print("[-] Unable to save new annotations!")

def generate_synthetic_data(pipe, valid_annot_data, category, prompt_dict, image_dir_path, mask_dir_path, synthetic_annotations, save_image_dir, save_annotations_dir, bg_height=720, bg_width=720, num_images=1):
    if num_images > 5:
        print("[+] Generation above 5 image not supported due to time constraints!")
    
    # Load data from relevant directories
    image_names = os.listdir(image_dir_path)
    image_mask_names = os.listdir(mask_dir_path)
    
    # Delete the .DS_Store file
    if '.DS_Store' in image_names:
        indx = image_names.index('.DS_Store')
        image_names.pop(indx)

    if '.DS_Store' in image_mask_names:
        indx = image_mask_names.index('.DS_Store')
        image_mask_names.pop(indx)

    # Let's generate num_images number of backgrounds
    for i in range(num_images):
        print("[+] I: ", i)
        # Make a random choice in the respective category
        choice = np.random.randint(low=0, high=len(image_names) - 1, size=1)[0]

        # Generate the background corresponding to the respective category
        image_name = image_names[choice]
        backgrounds_arr = generate_background_image(pipe, category, image_name, prompt_dict, 1)

        # Convert the background to the numpy arr
        background_image = backgrounds_arr[0]
        background_image = np.array(background_image)
        
        # Now load the image and the corresponding mask
        image_path = f"{image_dir_path}/{image_name}"
        mask_path = f"{mask_dir_path}/{image_name[:-4]}.png"

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        # Convert the mask to boolean mask
        mask = create_boolean_mask(mask)

        # Now we resize the background_image
        background_image = cv2.resize(background_image, (bg_width, bg_height))

        # Let's create a composite mask, i.e, mask for the entire composite image that will be create
        composite_mask = np.ones((background_image.shape[0], background_image.shape[1], 3))
        y_loc, x_loc = int(background_image.shape[0] / 2), int(background_image.shape[1] / 2)

        # Let's add the background to the image and generate the overall composite mask as well
        composite_image, composite_mask, _ = add_bg_to_image(image, mask, composite_mask, background_image, x_loc, y_loc)
        
        # Let's generate the synthetic keypoints data
        # Find the annotations corresponding to the image
        annot_data = find_image_annotations(valid_annot_data, image_name)
        if len(annot_data) == 0:
            print("[+] Image annotations not found try again!")


        # Let's add the synthetic keypoints to our data
        unique_composite_image_name = annot_data['name'][:-4] + "_" + str(datetime.datetime.now()) + ".jpg"
        save_annot_data = generate_synthetic_keypoints(unique_composite_image_name, annot_data, synthetic_annotations, x_loc, y_loc, image.shape[0], image.shape[1])

        # Save generate synthetic images and annotations
        save_image_path = f"{save_image_dir}/{unique_composite_image_name}"
        save_annotations_path = f"{save_annotations_dir}/synthetic_annotations.json"
        
        print(save_image_path, type(save_image_path))

        try:
            save_image(composite_image, save_image_path)
            print(f"[+] Synthetic image saved to {save_image_path}")
        except Exception as e:
            print(e)
            print("[+] Unable to save synthetic image, exiting the process")
            return
        
        try:
            save_annotations(save_annot_data, save_annotations_path)
            print(f"[+] Synthetic annotations save to {save_annotations_path}")
        except Exception as e:
            print(e)
            print("[+] Unable to save the synthetic annotations, exiting the process")
            # Do one more thing that is to delete the respective image, since process failed
            return
        print("[+] Synthetic data saved successfully!")
    return

def randomly_generate_synthetic_data(pipe, valid_annot_data, prompt_dict, num_data, category_dir_path, save_image_dir, save_annotations_dir):
    categories = os.listdir(category_dir_path)

    if '.DS_Store' in categories:
        indx = categories.index('.DS_Store')
        categories.pop(indx)
    
    for i in range(num_data):
        category_choice = random.randint(0, 19)
        category = categories[category_choice]
        
        print(f"[+] Generating an image belonging to {category} as random choice: {category_choice}")
        image_dir_path, mask_dir_path = f'{category_dir_path}/{category}/images', f'{category_dir_path}/{category}/mask'
        synthetic_annotations = []
        generate_synthetic_data(pipe, valid_annot_data, category, prompt_dict, image_dir_path, mask_dir_path, synthetic_annotations, save_image_dir, save_annotations_dir, 720, 720, 1)
        print("------------------------------------------------")
