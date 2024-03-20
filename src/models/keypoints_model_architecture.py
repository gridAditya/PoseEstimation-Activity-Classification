import os
import shutil
from ultralytics import YOLO

class KeypointsEstimationModel:
    def __init__(self, path):
        self.path = path
        self.model = YOLO(self.path)
        self.save_logs_path = None
        self.latest_logs_path = None
        print(f"[+] Model instance generated successfully from {self.path}")

    def train_model(self, config_path, hyperparameters):
        print("[+] Config File located, begginning the model training.")
        self.model.train(data=config_path, **hyperparameters)

    def save_logs(self, save_logs_path):
        self.latest_logs_path = './runs/pose'
        dirs = [os.path.join(self.latest_logs_path, dir) for dir in os.listdir(self.latest_logs_path) if os.path.isdir(os.path.join(self.latest_logs_path, dir))]
        self.save_logs_path = save_logs_path # save_log_path = ../../logs/keypoint_estimation_yolov8/pose

        most_recent_directory = max(dirs, key=os.path.getmtime)
        source_path = most_recent_directory
        destination_path = self.save_logs_path

        shutil.copytree(source_path, destination_path)

        print(f"[+] Source_path: {source_path} destination_path: {destination_path}")
        print(f"[+] Logs saved successfully")
