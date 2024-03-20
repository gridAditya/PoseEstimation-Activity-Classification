import sys
sys.path.append('src')

from models.keypoints_model_architecture import KeypointsEstimationModel

# Let's firstly create the model
model_obj = KeypointsEstimationModel('../../yolov8n-pose.pt')

# Now let's train the model
hyperparameters = {
    'epochs':25,
    'batch':64,
    'imgsz':(640, 480),
    'cache':True,
    'dropout':0.1,
    'cos_lr': True, 
    'close_mosaic':20,
    'freeze':21,
    'mosaic':0.0,
    'hsv_h':0.02,
    'hsv_s':0.7,
    'hsv_v':0.5,
    'translate':0.0,
    'scale':0.0,
    'degrees':0
}

model_obj.train_model('./src/models/config.yaml', hyperparameters) # path of the config.yaml file

model_obj.save_logs('./logs/keypoint_estimation_yolov8/pose') # save_logs path