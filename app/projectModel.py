from modelsReq.violence.model import Model
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from modelsReq.density.src.models.CSRNet import CSRNet
from modelsReq.yoloModels.objectCounter import ObjectCounter
from ultralytics import YOLO
from modelsReq.yoloModels.tracker import Tracker
import pandas as pd
import math
import threading
import os 
import cvzone
import pprint
from sklearn.cluster import KMeans
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
modelClothes = None
modelFaces = None
modelAgedetection = None

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
    global modelCrowd, model, modelV, modelDEns, modelG , modelClothes, modelFaces, modelAgedetection
    with model_lock:
        if modelCrowd is None:
            modelCrowd = YOLO('app/modelsReq/yoloModels/best_crowded.pt')

        if model is None:
            model = YOLO('app/modelsReq/yoloModels/yolov8m.pt')

        if modelV is None:
            modelV = Model()

        if modelDEns is None:
            modelDEns = CSRNet()

        if modelG is None:
            modelG = YOLO("app/modelsReq/yoloModels/genderV2.pt")

        if modelFaces is None :
            modelFaces = YOLO('app/modelsReq/yoloModels/yolov8n-face.pt')

        if modelClothes is None:
            modelClothes = YOLO('app/modelsReq/yoloModels/best.pt')

        if modelAgedetection is None:
            modelAgedetection = YOLO('app/modelsReq/yoloModels/yolo_age_detection.pt')




initialize_models()
#____________________________________________________________

region_points = [(20, 250), (680, 250), (680, 400), (20, 400)]
classes_to_count = [0]  

counter = ObjectCounter()
counter.set_args(view_img=False,
                 line_thickness=2,
                 reg_pts=region_points,
                 classes_names=model.names,
                 draw_tracks=True)


global x_in_out
x_in_out = 0

def enterExitCounting(frame , cameraname):
    tracks = model.track(frame, persist=True, show=False, verbose=False,
                         classes=classes_to_count)
    frame = counter.start_counting(frame, tracks)
    inn = counter.in_counts 
    out = counter.out_counts
    current = inn - out
    count = {
        'In' : out,
        'Out' : inn,
        'current' : abs(current)
    }

    global x_in_out
    x_in_out += 1
    dirr = f'app/output/in_out/{cameraname}'
    path = os.path.join(dirr, f'figure{x_in_out}.jpg')
    directory_path = os.path.dirname(path) 
    directory_path = create_directory_if_not_exists(directory_path)
    cv2.imwrite(path, frame)
    return path , count
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

def crowded(frame , cameraname):
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
    dirr = f'app/output/crowded/{cameraname}'
    path = os.path.join(dirr, f'figure{x_crowd}.jpg')
    directory_path = os.path.dirname(path) 
    directory_path = create_directory_if_not_exists(directory_path)
    cv2.imshow('frame' , frame)
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
    cv2.imshow('frame' , frame)
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

    bbox_list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        confidence = row[4]
        d = int(row[5])
        c = class_list[d]
        if ('car' in c) or ('truck' in c) or ('bus' in c) or ('bicycle' in c) or ('motorcycle' in c):
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
    x_vehicle = x_vehicle + 1
    dirr = f'app/output/vehicle/{cameraname}'
    path = os.path.join(dirr, f'figure{x_vehicle}.jpg')
    directory_path = os.path.dirname(path) 
    directory_path = create_directory_if_not_exists(directory_path)
    cv2.imshow('frame' , frame)
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
    path = ''
    x_violence = x_violence + 1
    if label in ['violence in office', 'fight on a street', 'street violence']:
        label = 'Violence'
        cv2.putText(frame, f'This is a {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 20, 255), 3)
        # cv2.imshow('frame',frame)
        dirr = f'app/output/violence/{cameraname}'
        path = os.path.join(dirr, f'figure{x_violence}.jpg')
        directory_path = os.path.dirname(path) 
        directory_path = create_directory_if_not_exists(directory_path)
        cv2.imwrite(path, frame)
    else :    
        label = 'No Voilence'
        cv2.putText(frame, f'This is a {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100,0), 3)
        # cv2.imshow('frame',frame)
        dirr = f'app/output/violence/{cameraname}'
        path = os.path.join(dirr, f'figure{x_violence}.jpg')
        directory_path = os.path.dirname(path) 
        directory_path = create_directory_if_not_exists(directory_path)
        cv2.imwrite(path, frame)        
    cv2.imshow('Image',frame)
    print(path,label)
    return path, label
#____________________________________________________________
global x_gender
x_gender = 0

#____________________________________________________________
def predict_Gender(frame):
    try:
        results = modelG(frame, stream=True)
        label_list = []
        for info in results:
            boxes = info.boxes
            for box in boxes:
                confidence = math.ceil(box.conf[0] * 100)
                if confidence >= 25:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = modelG.names[int(box.cls[0])]
                    dic = {'Gender': label, 'Confidence': confidence}
                    label_list.append(dic)
        return label_list
    except Exception as e:
        print(f'>> Error in gender detection: {str(e)}')
        return []

#____________________________________________________________
def detect_GENDER (frame, cameraname):
    global x_gender
    gender_info = []
    path = ''
    try:
        # Assign image to model
        face_result = modelFaces.predict(frame)
        for info in face_result:
            parameters = info.boxes
            for box in parameters:
                confidence = math.ceil(box.conf[0] * 100)
                if confidence >= 40:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1, x2, y2 = x1-10, y1-20, x2+10, y2+10

                    # Call gender detection function
                    label_list = predict_Gender(frame[y1:y2, x1:x2])

                    # Add gender info to the list
                    if label_list:
                        gender_info.append({'Person ID': len(gender_info) + 1,
                                            'Gender': label_list[0]['Gender'],
                                            'Confidence': label_list[0]['Confidence'],
                                            'Location': (x1, y1, x2, y2)})

        # Draw rectangles and text on the frame
        for info in gender_info:
            # Draw rectangle over the face
            cv2.rectangle(frame, (info['Location'][0], info['Location'][1]), 
                          (info['Location'][2], info['Location'][3]), (255, 100, 0), 4)
          
            # Display gender label and confidence inside the rectangle
            gender_text = f"{info['Gender']} : {info['Confidence']}%"
            cv2.putText(frame, gender_text, (info['Location'][0], info['Location'][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 255), 2)
            del info['Confidence']
        
            del info['Location']
        cv2.imshow('Image' , frame)
            
        if len(gender_info) != 0 :
            # Save the frame with drawn rectangles and text
            dirr = f'app/output/gender/{cameraname}'
            path = os.path.join(dirr, f'frame_{x_gender}.jpg')
            directory_path = os.path.dirname(path)
            directory_path = create_directory_if_not_exists(directory_path)
            cv2.imwrite(path, frame)
            x_gender += 1
        
        print(path,gender_info)           
        return path, gender_info

    except Exception as e:
        print("Error in face detection:", e)
        return path, []

#____________________________________________________________
def get_dominant_color_with_name(frame):

    """
    Extracts the dominant color and its name from a given frame.

    Args:
    - frame: Input frame in BGR format.

    Returns:
    - dominant_color_array: List containing RGB values of the dominant color.
    - color_name: Name of the dominant color.
    """
        
    # Convert frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Reshape the frame to be a list of pixels
    pixels = frame_rgb.reshape((-1, 3))

    # Use KMeans to find dominant color
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(pixels)

    # Get the dominant color
    dominant_color = kmeans.cluster_centers_[0]

    # Convert RGB to HSV
    hsv_value = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_RGB2HSV)[0][0]
    # Define expanded HSV color ranges
    color_ranges = {
        'Red': ([0, 100, 100], [10, 255, 255]),        # Red color range
        'Blue': ([94, 80, 2], [126, 255, 255]),        # Blue color range
        'Green': ([35, 52, 72], [85, 255, 255]),       # Green color range
        'Yellow': ([15, 100, 100], [35, 255, 255]),    # Yellow color range
        'Purple': ([125, 50, 50], [175, 255, 255]),    # Purple color range
        'Cyan': ([85, 100, 100], [105, 255, 255]),     # Cyan color range
        'Orange': ([5, 100, 100], [25, 255, 255]),     # Orange color range
        'Brown': ([0, 70, 70], [40, 255, 255]),        # Brown color range
        'Black': ([0, 0, 0], [179, 255, 30]),          # Black color range
        'White': ([0, 0, 230], [179, 30, 255])         # White color range
    }

    # Initialize variables for minimum distance and color name
    min_distance = float('inf')
    min_color_name = "Unknown"

    # Iterate through color ranges
    for color, (lower, upper) in color_ranges.items():
        if np.all(hsv_value >= lower) and np.all(hsv_value <= upper):
            return list(dominant_color.astype(int)), color

        # Calculate the distance between the given color and the color range
        distance = np.linalg.norm(hsv_value - np.mean([lower, upper], axis=0))
        # Check if this distance is smaller than the current minimum distance
        if distance < min_distance:
            min_distance = distance
            min_color_name = color

    return list(dominant_color.astype(int)), min_color_name


# def clothes(frame):
    """
    Detects and analyzes clothing attributes of objects within a given frame.

    Args:
        frame: Input frame containing objects.

    Returns:
        frame: Modified frame with bounding boxes and annotations.
        label_info_list: List containing dictionaries of detected objects with clothing attributes.
    """
    try:
        results = modelClothes(frame, stream=True)
        label_info_list = []

        for info in results:
            boxes = info.boxes
            for box in boxes:
                label_info = {}
                confidence = math.ceil(box.conf[0] * 100)
                if confidence >= 10:
                    x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 4)
                    label = model.names[int(box.cls[0])]
                    label_info['label'] = label
                    cv2.putText(frame, f"{label}: {confidence}%", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 255), 2)                        
                    roi = frame[y1:y2, x1:x2] 
                    dominant_color_array, color_name = get_dominant_color_with_name(roi)
                    label_info['Dominant Color Array'] = dominant_color_array
                    label_info['Dominant Color Name'] = color_name
                    label_info_list.append(label_info)

        return frame, label_info_list

    except Exception as e:
        print('\t>> Error:', e)
        return frame, []


global x_clo
x_clo = 0
def clothes(frame):
    try:
        results = modelClothes(frame, stream=True)
        label_info_list = []

        for info in results:
            boxes = info.boxes
            for box in boxes:
                label_info = {}
                confidence = math.ceil(box.conf[0] * 100)
                if confidence >= 25:
                    x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 4)
                    label = modelClothes.names[int(box.cls[0])]
                    label_info['label'] = label
                    cv2.putText(frame, f"{label}: {confidence}%", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 255), 2)
                    roi = frame[y1:y2, x1:x2]
                    dominant_color_array, color_name = get_dominant_color_with_name(roi)

                    # Convert dominant_color_array to list of integers
                    dominant_color_array = [int(val) for val in dominant_color_array]
                    label_info['Dominant Color Array'] = dominant_color_array
                    label_info['Dominant Color Name'] = color_name
                    label_info_list.append(label_info)

        return frame, label_info_list

    except Exception as e:
        print('\t>> Error:', e)
        return frame, []


def human_clothing(frame, cameraname):
    global x_clo
    results = model.predict(frame)
    px = pd.DataFrame(results[0].boxes.data).astype("float")
    list = []
    total = 0
    person_info_list = []
    path = ""
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


    bbox_id = tracker.update(bbox_list)
    for bbox in bbox_id:
        person_info = {}
        x3, y3, x4, y4, _ = bbox
        total += 1
        person_info['Person ID'] = total
        cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 100, 0), 2)
        person_roi = frame[y3:y4, x3:x4]
        clothes_frame, label = clothes(frame=person_roi)
        if len(label) == 0 :   person_info['clothes'] = 'Clothes Cannot Recognized'
        else :person_info['clothes'] = label
        
        person_info_list.append(person_info)

        cv2.putText(frame, f'Person {total}', (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 255), 2)
        cv2.imshow('frame' , frame)
        x_clo += 1
        dirr = f'app/output/clothes/{cameraname}'
        path = os.path.join(dirr, f'figure{x_clo}.jpg')
        directory_path = os.path.dirname(path) 
        directory_path = create_directory_if_not_exists(directory_path)
        cv2.imwrite(path, frame)

    cvzone.putTextRect(frame, f'Total Person: {total}', (50, 60), 2, 2, colorR=(0, 0, 255))
    pprint.pprint(person_info_list)
    return  path, person_info_list


#____________________________________________________________
#yolo Age_detection
x_age = 0
def predict_Age(frame):
    try:
        results = modelAgedetection(frame, stream=True)
        label_list = []
        for info in results:
            boxes = info.boxes
            for box in boxes:
                confidence = math.ceil(box.conf[0] * 100)
                if confidence >= 25:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = modelAgedetection.names[int(box.cls[0])]
                    dic = {'Age': label, 'Confidence': confidence}
                    label_list.append(dic)
        return label_list
    except Exception as e:
        print(f'>> Error in Age detection: {str(e)}')
        return []

#____________________________________________________________
def Age_detection (frame, cameraname):
    global x_age
    age_info = []
    path = ''
    try:
        # Assign image to model
        face_result = modelFaces.predict(frame)
        for info in face_result:
            parameters = info.boxes
            for box in parameters:
                confidence = math.ceil(box.conf[0] * 100)
                if confidence >= 40:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1, x2, y2 = x1-10, y1-20, x2+10, y2+10

                    # Call Age detection function
                    label_list = predict_Age(frame[y1:y2, x1:x2])

                    # Add Age info to the list
                    if label_list:
                        # if names[int(c)] =="0-5" or names[int(c)] =="6-10"or names[int(c)] == "11-15":
                        if label_list[0]['Age'] <= "11-15":   
                            age_info.append({'Person ID': len(age_info) + 1,
                                            'Age': "Young",
                                            'Confidence': label_list[0]['Confidence'],
                                            'Location': (x1, y1, x2, y2)})
                        else:
                            age_info.append({'Person ID': len(age_info) + 1,
                                            'Age': "Old",
                                            'Confidence': label_list[0]['Confidence'],
                                            'Location': (x1, y1, x2, y2)})


        # Draw rectangles and text on the frame
        for info in age_info:
            # Draw rectangle over the face
            cv2.rectangle(frame, (info['Location'][0], info['Location'][1]), 
                          (info['Location'][2], info['Location'][3]), (255, 100, 0), 4)
          
            # Display gender label and confidence inside the rectangle
            age_text = f"{info['Age']} "
            cv2.putText(frame, age_text, (info['Location'][0], info['Location'][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 255), 2)
            del info['Confidence']
        
            del info['Location']
        cv2.imshow('Image' , frame)
            
        if len(age_info) != 0 :
            # Save the frame with drawn rectangles and text
            dirr = f'app/output/Age/{cameraname}'
            path = os.path.join(dirr, f'frame_{x_age}.jpg')
            directory_path = os.path.dirname(path)
            directory_path = create_directory_if_not_exists(directory_path)
            cv2.imwrite(path, frame)
            x_age += 1
        
        print(path,age_info)           
        return path, age_info

    except Exception as e:
        print("Error in face detection:", e)
        return path, []
