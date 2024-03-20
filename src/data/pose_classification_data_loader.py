import sys
sys.path.append('src')

import tensorflow as tf
import numpy as np

# Function to load data from npy file
def load_data(file_path):
    return tf.cast(np.load(file_path), tf.float32)

# Function to load label from text file
def load_label(file_path):
    with open(file_path, 'r') as file:
        label = int(file.read().strip())
    return tf.cast(label, tf.int32)

def data_loader(dataset_path, x_train, y_train, x_val, y_val, x_train_suffix, y_train_suffix, x_val_suffix, y_val_suffix):

    # Generating an an array of file-paths for train and val dataset
    for i in range(len(x_train)):
        x_train[i] = dataset_path + '/' + x_train_suffix + '/' + x_train[i]
        y_train[i] = dataset_path + '/' + y_train_suffix + '/' + y_train[i]

    for i in range(len(x_val)):
        x_val[i] = dataset_path + '/' + x_val_suffix + '/' + x_val[i]
        y_val[i] = dataset_path + '/' + y_val_suffix + '/' + y_val[i]

    # Just for checking purposes
    print(f"[+] Train=> len_X: {len(x_train)} len_Y: {len(y_train)}")
    print(f"[+] Val=> len_X: {len(x_val)} len_Y: {len(y_val)}")

    # Creating a dataset instance
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    # Defining on how to load dataset during runtime
    train_dataset = train_dataset.map(lambda x_file, y_file: (tf.numpy_function(load_data, [x_file], tf.float32),
                                              tf.numpy_function(load_label, [y_file], tf.int32)))
    
    val_dataset = val_dataset.map(lambda x_file, y_file: (tf.numpy_function(load_data, [x_file], tf.float32),
                                              tf.numpy_function(load_label, [y_file], tf.int32)))
    

    data_shape=(480, 640, 4)
    # Defining the shape of the loaded data
    train_dataset = train_dataset.map(lambda x, y: (tf.ensure_shape(x, data_shape), tf.ensure_shape(y, ())))
    val_dataset = val_dataset.map(lambda x, y: (tf.ensure_shape(x, data_shape), tf.ensure_shape(y, ())))

    # Batching and shuffling the data
    train_dataset = train_dataset.shuffle(buffer_size=len(x_train)).batch(32)
    val_dataset = val_dataset.shuffle(buffer_size=len(x_val)).batch(32)

    return [train_dataset, val_dataset]

