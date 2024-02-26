from modelsReq.violence.model import Model
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from modelsReq.density.src.models.CSRNet import CSRNet
from ultralytics import YOLO
from modelsReq.yoloModels.tracker import Tracker
import pandas as pd
import math
import threading
import os 
import warnings
warnings.filterwarnings("ignore")

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory

# Initialize global variables
modelCrowd = None
model = None
modelV = None
modelDEns = None
modelG = None
#____________________________________________________________
# Lock for thread safety
model_lock = threading.Lock()
count = 0
#____________________________________________________________
# Load class list from file
my_file = open("app/modelsReq/yoloModels/coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
tracker = Tracker()
#____________________________________________________________
def initialize_models():
    """
    Initialize machine learning models.
    """
    global count
    count += 1
    print("Initializing models for the {} time".format(count))
    global modelCrowd, model, modelV, modelDEns, modelG
    with model_lock:
        if modelCrowd is None:
            modelCrowd = YOLO('app/modelsReq/yoloModels/best_crowded.pt')

        if model is None:
            model = YOLO('app/modelsReq/yoloModels/yolov8s.pt')

        if modelV is None:
            modelV = Model()

        if modelDEns is None:
            modelDEns = CSRNet()

        if modelG is None:
            modelG = YOLO('app/modelsReq/yoloModels/gender.pt')

initialize_models()
#____________________________________________________________
global x_density
x_density = 0
def crowdedDensity(frame  , cameraname):
    """
    Detect crowded density in a given frame.

    Args:
        frame (numpy.ndarray): Input frame.

    Returns:
        tuple: Tuple containing path to saved figure and count of people.
    """
    global x_density

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255
    frame = torch.from_numpy(frame).permute(2, 0, 1)

    # Predict
    predict = modelDEns(frame.unsqueeze(0))
    count = predict.sum().item()

    # Plot the results using Matplotlib
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 3))
    ax0.imshow(frame.permute(1, 2, 0))
    ax1.imshow(predict.squeeze().detach().numpy(), cmap='jet')
    ax0.set_title('People Count')
    ax1.set_title(f'People Count = {count:.0f}')
    ax0.axis("off")
    ax1.axis("off")
    plt.tight_layout()

    # Save the figure
    x_density = x_density + 1

    dirr = f'app/output/density/{cameraname}'
    path = os.path.join(dirr, f'figure{x_density}.jpg')
    directory_path = os.path.dirname(path) 
    directory_path = create_directory_if_not_exists(directory_path)
    plt.savefig(path)
    plt.close()
    print('Figure saved successfully.')

    return path, count
#____________________________________________________________
global x_crowd
x_crowd = 0

def crowded(frame):
    """
    Detect crowded areas in a given frame.

    Args:
        frame (numpy.ndarray): Input frame.

    Returns:
        tuple: Tuple containing path to saved figure and count of people.
    """
    global x_crowd
    count = 0
    results = modelCrowd(frame, stream=True)

    for info in results:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence >= 40:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                count += 1

    cv2.putText(frame, f"Count : {count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 5)
    x_crowd = x_crowd + 1
    path = f'app/output/crowded/figure{x_crowd}.jpg'
    cv2.imwrite(path, frame)

    return path, count
#____________________________________________________________
global x_crossing
x_crossing = 0
def crossingBorder(frame,cameraname):
    """
    Detect people crossing a border in a given frame.

    Args:
        frame (numpy.ndarray): Input frame.

    Returns:
        tuple: Tuple containing path to saved figure and count of people crossing the border.
    """
    global x_crossing
    count = 0
    results = model.predict(frame)

    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    bbox_list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'person' in c:
            bbox_list.append([x1, y1, x2, y2])
            count += 1

    bbox_id = tracker.update(bbox_list)
    for bbox in bbox_id:
        x3, y3, x4, y4, d = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2
        cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)

    cv2.putText(frame, f'Count: {count}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    x_crossing += 1
    dirr = f'app/output/crossing/{cameraname}'
    path = os.path.join(dirr, f'figure{x_crossing}.jpg')
    directory_path = os.path.dirname(path) 
    directory_path = create_directory_if_not_exists(directory_path)

    cv2.imwrite(path, frame)

    return path, count
#____________________________________________________________
global x_vehicle
x_vehicle = 0
def vehicleCounting(frame , cameraname):
    """
    Detect vehicles in a given frame.

    Args:
        frame (numpy.ndarray): Input frame.

    Returns:
        tuple: Tuple containing path to saved figure and count of vehicles.
    """
    global x_vehicle
    count = 0
    results = model.predict(frame)

    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    l = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c or 'truck' in c or 'bus' in c or 'bicycle' in c or 'motorcycle' in c:
            list.append([x1, y1, x2, y2])
            count += 1

    bbox_id = tracker.update(l)
    for bbox in bbox_id:
        x3, y3, x4, y4, d = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2
        cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)

    cv2.putText(frame, f'Count: {count}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    x_vehicle = x_vehicle + 1
    dirr = f'app/output/vehicle/{cameraname}'
    path = os.path.join(dirr, f'figure{x_vehicle}.jpg')
    directory_path = os.path.dirname(path) 
    directory_path = create_directory_if_not_exists(directory_path)
    cv2.imwrite(path, frame)

    return path, count
#____________________________________________________________
global x_violence
x_violence = 0
def violence(frame , cameraname):
    """
    Detect violence in a given frame.

    Args:
        frame (numpy.ndarray): Input frame.

    Returns:
        tuple: Tuple containing path to saved figure and label indicating presence of violence.
    """
    global x_violence
    RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictions = modelV.predict(image=RGBframe)
    label = predictions['label']
    if label in ['violence in office', 'fight on a street', 'street violence']:
        label = 'Predicted Violence'
    cv2.putText(frame, f'This is a {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    x_violence = x_violence + 1
    path = ''
    if label == 'Predicted Violence':
        dirr = f'app/output/violence/{cameraname}'
        path = os.path.join(dirr, f'figure{x_violence}.jpg')
        directory_path = os.path.dirname(path) 
        directory_path = create_directory_if_not_exists(directory_path)
        cv2.imwrite(path, frame)

    return path, label
#____________________________________________________________
global x_gender
x_gender = 0
def detect_GENDER(frame , cameraname):
    """
    Detect gender in a given frame.

    Args:
        frame (numpy.ndarray): Input frame.

    Returns:
        tuple: Tuple containing path to saved figure and detected gender label.
    """
    global x_gender
    try:
        results = modelG(frame, stream=True)

        for info in results:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                if confidence >= 40:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 4)
                    label = modelG.names[int(Class)]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    font_thickness = 2
                    text_color = (0, 120, 255)
                    cv2.putText(frame, f"{label}: {confidence}%", (x1, y1 - 10),
                                font, font_scale, text_color, font_thickness)

                    x_gender = x_gender + 1
                    dirr = f'app/output/gender/{cameraname}'
                    path = os.path.join(dirr, f'figure{x_gender}.jpg')
                    directory_path = os.path.dirname(path) 
                    directory_path = create_directory_if_not_exists(directory_path)
                    cv2.imwrite(path, frame)

        return path, label

    except Exception as e:
        print(f'>> Error: {str(e)}')
        return None, None


