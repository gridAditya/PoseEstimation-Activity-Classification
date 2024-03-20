import os
import copy

import math
import numpy as np
import json, yaml
import cv2
from matplotlib import pyplot as plt
from pymatreader import read_mat
from typing import Any, List, Tuple, Union

def extract_valid_data_from_matlab_file(data, id, num_images, image_files_path):
    """
    Extracts the valid annotations from .mat file
    
    Args:
        data: dictionary containing .mat file data
        id: an integer array containing values from 0-15(inclusive) representing the various joints in human body
        num_images: an integer representing the total number of images in the dataset
        image_files_path: a string containing the relative path of the unprocessed images
    
    Return:
        valid_annot_data: an array containing the annotations corresponding to each image in the dataset
    """
    initial_index = 0
    batch = 32
    valid_annot_data = []
    
    while initial_index < num_images:
        for img_idx in range(initial_index, min(initial_index + batch, num_images)):
            img_name = data['annolist']['image'][img_idx]['name']
            img_path = image_files_path + '/' + img_name
            
            if not (os.path.isfile(img_path)):
                print(f"[-] File named {img_name} does not exist")
                continue

            # Dealing with potential causes of errors in determining num_valid_persons
            # An edge case observed here: # Indx: 3605 has data['annolist']['annorect'][3605] as None Type
            if data['annolist']['annorect'][img_idx] is None:
                continue
            if ((len(data['annolist']['annorect'][img_idx]) == 0)):
                continue

            if 'annopoints' not in data['annolist']['annorect'][img_idx]:
                continue
            if len(data['annolist']['annorect'][img_idx]['annopoints']) == 0:
                continue
            
            if 'objpos' not in data['annolist']['annorect'][img_idx]:
                continue

            # Determine Number of valid persons in the image
            if type(data['annolist']['annorect'][img_idx]['objpos']) == dict:
                num_valid_person = 1
            else:
                num_valid_person = len(data['annolist']['annorect'][img_idx]['objpos'])
            
            img_data = {
                'name': '',
                'idx': 0,
                'general_activity_name': '',
                'personInfo': []
            }
            
            annot_data = {
                'x1': 0,
                'x2': 0,
                'y1': 0,
                'y2': 0,
                'x':[],
                'y':[],
                'id': [],
                'is_visible': [],
            }

            img_data['name'] = img_name
            img_data['idx'] = img_idx

            if type(data['annolist']['annorect'][img_idx]['annopoints']) == dict:
                data['annolist']['annorect'][img_idx]['annopoints'] = [copy.deepcopy(data['annolist']['annorect'][img_idx]['annopoints'])]
            
            for i in range(num_valid_person):
                x, y, is_visible = [], [], []
                
                # Some Edge Cases observed:
                # 1: Some images have only one person like img_idx=8
                # 2: Img_idx = 2248: even if num=2, data['annolist']['annorect'][img_idx]['annopoints'][1] is an empty array
                # 3: Img_idx = 15007: incomplete annotations provided like only 6 joints instead of 16 joints
                # 4: Some idx = 24730 at[1] dont' have feilds like is_visible

                # Check if info related to 'x', 'y', 'id', 'visibility' of joints locations in image exists
                if len(data['annolist']['annorect'][img_idx]['annopoints'][i]) == 0: # [24732]....[0/1] 
                    continue
                if 'point' not in data['annolist']['annorect'][img_idx]['annopoints'][i]:
                    continue
                if 'x' not in data['annolist']['annorect'][img_idx]['annopoints'][i]['point']:
                    continue
                if 'y' not in data['annolist']['annorect'][img_idx]['annopoints'][i]['point']:
                    continue
                if 'id' not in data['annolist']['annorect'][img_idx]['annopoints'][i]['point']:
                    continue
                if 'is_visible' not in data['annolist']['annorect'][img_idx]['annopoints'][i]['point']:
                    continue
                
                # Convert 'x', 'y' co-ordinates of joints to a list of integers if they are not in list format
                if type(data['annolist']['annorect'][img_idx]['annopoints'][i]['point']['x']) == int: # some 'x' like [24730]..[1] has a single int entry
                    data['annolist']['annorect'][img_idx]['annopoints'][i]['point']['x'] = [copy.deepcopy(data['annolist']['annorect'][img_idx]['annopoints'][i]['point']['x'])]

                if type(data['annolist']['annorect'][img_idx]['annopoints'][i]['point']['y']) == int:
                    data['annolist']['annorect'][img_idx]['annopoints'][i]['point']['y'] = [copy.deepcopy(data['annolist']['annorect'][img_idx]['annopoints'][i]['point']['y'])]

                if type(data['annolist']['annorect'][img_idx]['annopoints'][i]['point']['id']) == int:
                    data['annolist']['annorect'][img_idx]['annopoints'][i]['point']['id'] = [copy.deepcopy(data['annolist']['annorect'][img_idx]['annopoints'][i]['point']['id'])]

                if type(data['annolist']['annorect'][img_idx]['annopoints'][i]['point']['is_visible']) == int:
                    data['annolist']['annorect'][img_idx]['annopoints'][i]['point']['is_visible'] = [copy.deepcopy(data['annolist']['annorect'][img_idx]['annopoints'][i]['point']['is_visible'])]
                
                # Gather information related to each person's bounding box
                if (('x1' not in data['annolist']['annorect'][img_idx]) or ('x2' not in data['annolist']['annorect'][img_idx])):
                    continue
                if (('y1' not in data['annolist']['annorect'][img_idx]) or ('y2' not in data['annolist']['annorect'][img_idx])):
                    continue

                if type(data['annolist']['annorect'][img_idx]['x1']) == int:
                    data['annolist']['annorect'][img_idx]['x1'] = [copy.deepcopy(data['annolist']['annorect'][img_idx]['x1'])]
                if type(data['annolist']['annorect'][img_idx]['x2']) == int:
                    data['annolist']['annorect'][img_idx]['x2'] = [copy.deepcopy(data['annolist']['annorect'][img_idx]['x2'])]

                if type(data['annolist']['annorect'][img_idx]['y1']) == int:
                    data['annolist']['annorect'][img_idx]['y1'] = [copy.deepcopy(data['annolist']['annorect'][img_idx]['y1'])]
                if type(data['annolist']['annorect'][img_idx]['y2']) == int:
                    data['annolist']['annorect'][img_idx]['y2'] = [copy.deepcopy(data['annolist']['annorect'][img_idx]['y2'])]

                # print("[-] Came till here....")
                # Gather information related to each person's joint location
                for jt_id in id:
                    if jt_id in data['annolist']['annorect'][img_idx]['annopoints'][i]['point']['id']:
                        idx = (data['annolist']['annorect'][img_idx]['annopoints'][i]['point']['id']).index(jt_id)
                        
                        if (idx < len(data['annolist']['annorect'][img_idx]['annopoints'][i]['point']['x'])) and (idx < len(data['annolist']['annorect'][img_idx]['annopoints'][i]['point']['y'])):
                            if (data['annolist']['annorect'][img_idx]['annopoints'][i]['point']['x'][idx] >= 0 and data['annolist']['annorect'][img_idx]['annopoints'][i]['point']['y'][idx] >= 0): # some images have negative entries
                                x.append(data['annolist']['annorect'][img_idx]['annopoints'][i]['point']['x'][idx])
                                y.append(data['annolist']['annorect'][img_idx]['annopoints'][i]['point']['y'][idx])
                            else:
                                x.append(0)
                                y.append(0)
                                is_visible.append(0)
                                print(f'[+] Img_name: {img_name} has a negative keypoint co-ordinates, its handled.')
                        else:
                            x.append(0)
                            y.append(0)
                            is_visible.append(0)
                        
                        # some 'is_visible' entries are empty np.ndarray, so we only consider entries that have int values
                        # some entries like 16882 have 'is_visible' entries as str
                        if idx < len(data['annolist']['annorect'][img_idx]['annopoints'][i]['point']['is_visible']):
                            if (type(data['annolist']['annorect'][img_idx]['annopoints'][i]['point']['is_visible'][idx]) == int):
                                is_visible.append(data['annolist']['annorect'][img_idx]['annopoints'][i]['point']['is_visible'][idx])
                            elif (type(data['annolist']['annorect'][img_idx]['annopoints'][i]['point']['is_visible'][idx]) == str):
                                is_visible.append(int(data['annolist']['annorect'][img_idx]['annopoints'][i]['point']['is_visible'][idx]))
                            else:
                                is_visible.append(0)
                        else:
                            is_visible.append(0)               
                    else:
                        x.append(0)
                        y.append(0)
                        is_visible.append(0)
                    
                # Store the information about each person in image in annot_data
                annot_data['x1'] = data['annolist']['annorect'][img_idx]['x1'][i]
                annot_data['x2'] = data['annolist']['annorect'][img_idx]['x2'][i]
                annot_data['y1'] = data['annolist']['annorect'][img_idx]['y1'][i]
                annot_data['y2'] = data['annolist']['annorect'][img_idx]['y2'][i]
                annot_data['x'] = copy.deepcopy(x)
                annot_data['y'] = copy.deepcopy(y)
                annot_data['id'] = copy.deepcopy(id)
                annot_data['is_visible'] = copy.deepcopy(is_visible)
                # Store the information about each image in img_data
                (img_data['personInfo']).append(copy.deepcopy(annot_data))
            
            # Store the category information
            img_data['general_activity_name'] = str(data['act']['cat_name'][img_idx])
            # Store the information related to all images in valid_annot_data
            valid_annot_data.append(img_data)
        initial_index = initial_index + batch
    return valid_annot_data

