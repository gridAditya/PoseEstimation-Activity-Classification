import os, shutil
import numpy as np
import torch, gzip

from torch.nn.utils import prune

def create_directory(directory_path):
    """
    Create a directory if it does not exist.

    Args:
        directory_path (str): The path of the directory to create.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def find_unstructured_prune_ratio(model):
    total_params = 0
    pruned_params = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and hasattr(module, 'weight'):
            total_params += module.weight.numel() # count the number of weights in the given layer
            pruned_params += (module.weight == 0).sum().item()  # Count the number of pruned parameters
    return [total_params, pruned_params, (pruned_params / total_params)]

def unstructured_prune_model(model, pruning_ratios):
    for name, module in model.named_modules():
        if ("cv1.conv" in name) or ("cv2.conv" in name):
            prune.l1_unstructured(module, name='weight', amount=pruning_ratios[0])  # Structured pruning along the first dimension (channels)
            prune.remove(module, 'weight')

        if ("cv2.1.0.conv" in name) or ("cv2.1.1.conv" in name) or ("cv2.2.0.conv" in name) or ("cv2.2.1.conv" in name): # (64, 64) and (64, 64)
            prune.l1_unstructured(module, name='weight', amount=pruning_ratios[1])  # Structured pruning along the first dimension (channels)
            prune.remove(module, 'weight')

        # Prune the c3 layer
        if ("cv3.0.0.conv" in name) or ("cv3.0.1.conv" in name):# # (64, 64) and  # (64, 64)
            prune.l1_unstructured(module, name='weight', amount=pruning_ratios[2])  # Structured pruning along the first dimension (channels)
            prune.remove(module, 'weight')
        
        if ("cv3.1.1.conv" in name): # (64, 64)
            prune.l1_unstructured(module, name='weight', amount=pruning_ratios[3])  # Structured pruning along the first dimension (channels)
            prune.remove(module, 'weight')
        
        if ("cv3.2.0.conv" in name): # (256, 64)
            prune.l1_unstructured(module, name='weight', amount=pruning_ratios[4])  # Structured pruning along the first dimension (channels)
            prune.remove(module, 'weight')
            
        if ("cv3.2.1.conv" in name): # (64, 64)
            prune.l1_unstructured(module, name='weight', amount=pruning_ratios[5])  # Structured pruning along the first dimension (channels)
            prune.remove(module, 'weight')
        
        # These c4 layer can be pruned equally (64ip, 48op) and (48ip, 48op)
        if ("cv4.0.0.conv" in name) or ("cv4.0.1.conv" in name):
            prune.l1_unstructured(module, name='weight', amount=pruning_ratios[6])  # Structured pruning along the first dimension (channels)
            prune.remove(module, 'weight')
        
        # Prune the c4.1.0 more(128ip, 48op) c4.1.1(48ip, 48op)
        if ("cv4.1.0.conv" in name):
            prune.l1_unstructured(module, name='weight', amount=pruning_ratios[7])  # Structured pruning along the first dimension (channels)
            prune.remove(module, 'weight')
        
        if ("cv4.1.1.conv" in name):
            prune.l1_unstructured(module, name='weight', amount=pruning_ratios[8])  # Structured pruning along the first dimension (channels)
            prune.remove(module, 'weight')
        
        # Prune the c4.2.0(256ip, 48op) more and c4.2.1(48ip, 48op) less
        if ("cv4.2.0.conv" in name):
            prune.l1_unstructured(module, name='weight', amount=pruning_ratios[9])  # Structured pruning along the first dimension (channels)
            prune.remove(module, 'weight')
        
        if ("cv4.2.1.conv" in name):
            prune.l1_unstructured(module, name='weight', amount=pruning_ratios[10])  # Structured pruning along the first dimension (channels)
            prune.remove(module, 'weight')
            
    print("[+] Pruning completed")
    return model

def get_subdirs_sorted_by_creation_time(directory):
    # Get all subdirectories in the given directory
    subdirs = [os.path.join(directory, name) for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    
    # Sort subdirectories by creation time
    subdirs_sorted = sorted(subdirs, key=lambda x: os.path.getctime(x))
    
    return subdirs_sorted

def copy_file(source_path, destination_path):
    try:
        shutil.copy(source_path, destination_path)
        print(f"File copied successfully from {source_path} to {destination_path}")
    except Exception as e:
        print("[-]Unable to copy file.")
        print(e)

def create_directory_if_not_exists(directory_path):
    # Check if the directory already exists
    if not os.path.exists(directory_path):
        # If it doesn't exist, create it recursively
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def is_train_dir(dir_name):
    if 'train' in dir_name:
        return True
    return False

def has_weights(dir_name):
    entries = os.listdir(dir_name)

    for entry in entries:
        if os.path.isdir(os.path.join(dir_name, entry)):
            if entry == 'weights':
                return True
    return False

def compress_file(input_file, output_file):
    try:
        with open(input_file, 'rb') as f_in:
            with gzip.open(output_file, 'wb') as f_out:
                f_out.writelines(f_in)
        print(f"File compressed successfully: {output_file}")
    except Exception as e:
        print(f"Unable to compress file. {e}")

def get_compressed_file_size(file_path):
    try:
        size = os.path.getsize(file_path)
        print(f"Size of compressed file '{file_path}': {size} bytes")
        return size
    except Exception as e:
        print(f"Unable to get compressed file size. {e}")

def measure_inference_speeds(model, entries, random_choices, dir_path):
    avg_speed = 0
    for i in range(len(random_choices)):
        image_path = dir_path + '/' + entries[random_choices[i]]

        # Run inference on original model
        result = model.predict(image_path)
        avg_speed += result[0].speed['inference']
    return [avg_speed/ len(random_choices)]

