import tensorflow as tf
import datetime

class PoseClassificationModel:
    def __init__(self):
        print("[+] Model creation started...")

        # Creating the model
        backbone = tf.keras.applications.EfficientNetV2B0(include_top=False)
        backbone.trainable = False # Freeze the layers of the backbone

        # Define the input layer
        input_layer = tf.keras.layers.Input(shape=(480, 640, 4), name="input_layer")

        x = tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), padding="same")(input_layer)
        x = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), padding="same")(x)

        x = backbone(x)
        x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)

        # Define the output layer
        ouput_layer = tf.keras.layers.Dense(20, activation="softmax", name="output_layer")(x)

        # Build the model
        self.model = tf.keras.Model(input_layer, ouput_layer)

        # Specify other attributes
        self.tensorboard_callback = None
        self.checkpoint_callback = None    

    def get_model_summary(self):
        return self.model.summary()
    
    def compile_model(self):
        self.model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=["accuracy"])
    
    def create_tensorboard_callback(self, dir_name, experiment_name):
        # dir_name: /logs/pose_classsification_logs_tensorflow_hub
        # name: pose_classification_based_on_keypoint_estimation
        log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir
        )
        return tensorboard_callback

    def create_checkpoint_callback(self, checkpoint_path):
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True, # set to False to save the entire model
                                                         save_best_only=True, # save only the best model weights instead of a model every epoch
                                                         save_freq="epoch", # save every epoch
                                                         verbose=1)
        return checkpoint_callback
    
    def train_model(self, train_dataset, val_dataset, epochs, tensorboard_callback_dir_path, tensorboard_callback_experiment_name, checkpoint_path):
        # Creating the callbacks
        self.tensorboard_callback = self.create_tensorboard_callback(tensorboard_callback_dir_path, tensorboard_callback_experiment_name)
        self.checkpoint_callback = self.create_checkpoint_callback(checkpoint_path)

        # Fitting the model
        history = self.model.fit(train_dataset,
                                 epochs=epochs,
                                 validation_data=val_dataset,
                                 callbacks=[self.tensorboard_callback, self.checkpoint_callback])
        
        # Return the history of the training
        return history

