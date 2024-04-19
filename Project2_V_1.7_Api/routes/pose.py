import cv2
import sys
import numpy as np
import tensorflow as tf
import hashlib, skimage
import secrets, jwt, json
import subprocess
from ultralytics import YOLO


from fastapi import APIRouter
from fastapi import Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from datetime import datetime, timedelta
from matplotlib import pyplot as plt

from models.users import User
from db.db_connect import connection
from utils.utils import verify_auth_token
from utils.model_utils import run_inference
from utils.model_utils import check_if_valid_image
from utils.utils import process_images
from utils.utils import calc_percent_of_major_shades

pose = APIRouter()

templates = Jinja2Templates(directory="templates")

# Get access to database
db = connection.pose

# Load the skeleton data
skeleton_data = None
skeleton_data_path = "./app_data/skeleton_data.json"
try:
    with open(skeleton_data_path, 'r') as f:
        skeleton_data = json.load(f)
    print("[+] Human skeleton data loaded successfully.")
except Exception as e:
    print("[-] Loading human skeleton data has failed. Exiting the process")
    sys.exit(1) # Exit the current process with status code 1

# Load the class_names
class_names = None
class_names_path = "./app_data/class_names.json"
try:
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    print("[+] Activity classes loaded successfully.")
except Exception as e:
    print("[-] Loading the activity classes has failed. Exiting the process")
    sys.exit(1) # Exit the current process with status code 1

# Specify the model paths
keypoints_model_path = './deep_learning_models/quantized.onnx'
pose_class_model_path = './deep_learning_models/pose_classification_13:14:15_18831.keras'
obj_detection_model_path = './deep_learning_models/yolov8m.pt'

# Load the models
keypoints_model = YOLO(keypoints_model_path, task='pose')
pose_classification_model = tf.keras.models.load_model(pose_class_model_path)
obj_detection_model = YOLO(obj_detection_model_path)

# Define the image_height and image_width
image_height, image_width = 480, 640

# GET REQUEST START-----------------------------------------------------------------------------------------
@pose.get("/", response_class=HTMLResponse)
async def welcome_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@pose.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@pose.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@pose.get("/evaluate_pose.html", response_class=HTMLResponse)
async def pose_evaluation_page(request: Request):
    return templates.TemplateResponse("send_data.html", {"request": request})

# POST REQUEST START-----------------------------------------------------------------------------------------
@pose.post("/signup")
async def create_user(request: Request):
    form_data = await request.form()
    username = (form_data["username"]).lower().strip()
    email = (form_data["email"]).lower().strip()
    password = form_data["password"]
    confirm_password = form_data["confirm_password"]


    if (password != confirm_password):
        raise HTTPException(status_code=400, detail={"error": "Incorrect password"})
    
    # Generate salt
    salt = secrets.token_hex(16)

    # Hash password with salt
    hashed_password = hashlib.sha256((password + salt).encode()).hexdigest()

    # Generate authentication token
    payload = {
        "email": email,
        "exp": datetime.now() + timedelta(days=1)  # Expiration time (e.g., 1 day)
    }
    secret_key = "magatsuKami"
    token = jwt.encode(payload, secret_key, algorithm="HS256")

    # Get current timestamp
    timestamp = datetime.now()
    try:
        user = User(
            username=username,
            email=email,
            password=hashed_password,
            salt=salt,
            tokens=[token],
            request_count=0,
            last_request_timestamp=timestamp
        )
        
        # Check if another user with same email already exists
        doc_matches = db.users.find_one({"email": email})
        if (doc_matches):
            print("[-] User already exists...")
            raise HTTPException(status_code=400, detail={"error": "One or more entries are invalid!"})
        
        # Create the user
        user = db.users.insert_one(dict(user))
        return {
                "message": "user created successfully",
                "token": token
            }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail={"error": "One or more entries are invalid!"})

@pose.post("/login")
async def login_user(request: Request):
    form_data = await request.form()

    # Extract user-info
    email = (form_data["email"]).lower().strip()
    password = form_data["password"]
    try:
        # Check whether the user exists or not
        user = db.users.find_one({"email": email})
        if not user:
            raise HTTPException(status_code=400, detail={"error": "One or more entries are invalid!"})
        
        # Match the password(hash + salt)
        salt = user['salt']
        hashed_password = hashlib.sha256((password + salt).encode()).hexdigest()
        true_password_hash = user['password']

        if hashed_password != true_password_hash:
            raise HTTPException(status_code=400, detail={"error": "One or more entries are invalid!"})
        
        # Create a new session token and append it to the array
        payload = {
            "email": email,
            "exp": datetime.now() + timedelta(days=1)  # Expiration time (e.g., 1 day)
        }
        secret_key = "magatsuKami"
        token = jwt.encode(payload, secret_key, algorithm="HS256")

        # Update the user
        db.users.update_one(
            {"email": email},
            {"$push": {"tokens": token}}
        )
        print("[+] Logged in successfully")
        return {"token": token}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail={"error": "One or more entries are invalid!"})

@pose.post("/logout")
async def logout_current_session(request: Request):
    try:
        # Extract the token
        token = (request.headers['Authorization']).replace("Bearer ", "")
        
        # Get the email
        decoded_payload = jwt.decode(token, "magatsuKami", algorithms=["HS256"])
        email = decoded_payload["email"]
        
        # Find the user
        user = db.users.find_one({"email": email})

        # Delete the token
        tokens_arr = user['tokens']
        tokens_arr = [usr_token for usr_token in tokens_arr if usr_token != token]
        
        # Update the value
        result = db.users.update_one(
            {"email": email},
            {"$set": {"tokens": tokens_arr}}
        )
        if result.modified_count != 1:
            raise HTTPException(status_code=500, detail={"error": "Logging Out Failed. Please try again!"})
        return {"message": "Current User session deleted"}
    except Exception as e:
        print("[-] Internal Server Error")
        raise HTTPException(status_code=500, detail={"error": "A Server error has occured!"})

@pose.post("/logoutAll")
async def logout_all_sessions(request: Request):
    try:
        # Extract the token
        token = (request.headers['Authorization']).replace("Bearer ", "")
        
        # Get the email
        decoded_payload = jwt.decode(token, "magatsuKami", algorithms=["HS256"])
        email = decoded_payload["email"]
        
        # Find the user
        user = db.users.find_one({"email": email})

        # Delete the token
        tokens_arr = user['tokens']
        tokens_arr = []
        
        # Update the value
        result = db.users.update_one(
            {"email": email},
            {"$set": {"tokens": tokens_arr}}
        )
        if result.modified_count != 1:
            raise HTTPException(status_code=500, detail={"error": "Logging Out Failed. Please try again!"})
        return {"message": "Logged out from all active sessions."}
    except Exception as e:
        print("[-] Internal Server Error")
        raise HTTPException(status_code=500, detail={"error": "A Server error has occured!"})

@pose.post("/evaluate_pose/{item_id}")
async def evaluate_item(item_id: int, request: Request):
    # Verify the auth_token
    token = (request.headers['Authorization']).replace("Bearer ", "")
    payload = verify_auth_token(token)
   
    # Authorization Completed now move-onto logic
    form_data = await request.form()
    try:
        # Access data in "imageDescription" field
        image_description = form_data["imageDescription"]

        # Access file uploaded in "imageUpload" field
        image_upload = form_data["imageUpload"]

        # Get the filename
        image_name = image_upload.filename

        # Read the image data as bytes
        image_bytes = await image_upload.read()

        image = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Apply gaussian blur to reduce noise
        image = cv2.GaussianBlur(image, (3, 3), 1)

        # # Check if the image is low constrast
        is_low_contrast = skimage.exposure.is_low_contrast(image, fraction_threshold=0.35, lower_percentile=1, upper_percentile=99, method="linear")
        if is_low_contrast:
            raise HTTPException(status_code=401, detail={"error": "Low constrast image. Please upload another image!"})

        # Check for no-human like image
        results = obj_detection_model(source=image, show=False, conf=0.4)
        class_present = results[0].boxes.cls
        confidence_scores = results[0].boxes.conf

        if not check_if_valid_image(class_present, confidence_scores):
            raise HTTPException(status_code=401, detail={"error": "Image does not contain humans."})
        
        # Let's check the available shades in the image, reject the image if two major shades of color >= 65% img_area
        cv2.imwrite("image_req.jpg", image)

        # Spawn a new subprocess to run Kmeans clustering since if we run KMeans directly it blocks the entire application
        process = subprocess.Popen(
            ["python", "-c", "import sys, os, cv2, json; sys.path.append('.'); from utils.utils import run_kmeans; image=cv2.imread('image_req.jpg'); result=run_kmeans(10, image); print(result)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # Wait for the subprocess to finish and capture its output
        stdout, stderr = process.communicate()

        # Check if the subprocess returned an error
        if process.returncode != 0:
            # Handle subprocess error
            print("[-] Error has occured within the subprocess")
            print(stderr)
            return {"error": stderr}
        
        img_valid, centroid = calc_percent_of_major_shades()
        if not img_valid:
            raise HTTPException(status_code=401, detail={
                "error": "Two shades of color cover more than 65 percent of image-area. Please upload another image.",
                "type": "shade error",
                "centroid": centroid
                })
    
        # Run the inference on the models
        image, pose_image, label = run_inference(skeleton_data, keypoints_model, pose_classification_model, image, class_names, image_height, image_width)
        image_encoded, pose_image_encoded = process_images(image, pose_image)
        return JSONResponse(content={"image": image_encoded, "pose_image": pose_image_encoded, "label": label})
    except Exception as e:
        print(e)
        return {"error": e}

@pose.get("/test", response_class=HTMLResponse)
async def test_page(request: Request):
    return templates.TemplateResponse("test.html", {"request": request})

@pose.get("/test1", response_class=HTMLResponse)
async def test_page(request: Request):
    return templates.TemplateResponse("test1.html", {"request": request})
