import cv2
from ultralytics import YOLO
import numpy as np
import math
import torch
import os
import threading

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

modelCrowd = None
model = None
modelV = None
modelDEns = None
modelG = None
modelClothes = None
modelFaces = None


#____________________________________________________________
# Lock for thread safety
model_lock = threading.Lock()
count = 0
#____________________________________________________________
# Load class list from file
my_file = open("app/modelsReq/yoloModels/coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
#____________________________________________________________
def initialize_models():
    """
    Initialize machine learning models.
    """
    global count
    count += 1
    print("Initializing models for the {} time".format(count))
    global modelCrowd, model, modelV, modelDEns, modelG , modelClothes, modelFaces
    with model_lock:
        if modelCrowd is None:
            modelCrowd = YOLO('app/modelsReq/yoloModels/best_crowded.pt')

        if model is None:
            model = YOLO('app/modelsReq/yoloModels/yolov8m.pt')

        if modelG is None:
            modelG = YOLO("app/modelsReq/yoloModels/genderV2.pt")

        if modelFaces is None :
            modelFaces = YOLO('app/modelsReq/yoloModels/yolov8n-face.pt')

        if modelClothes is None:
            modelClothes = YOLO('app/modelsReq/yoloModels/best.pt')




initialize_models()

# Ensure that the YOLO model is configured for GPU
model = YOLO("app\modelsReq\yoloModels\genderV2.pt")
face_model = YOLO('app\modelsReq\yoloModels\yolov8n-face.pt')

# Global variable for frame count
global x_gender
x_gender = 0

def predict_Gender(frame):
    try:
        results = model(frame, stream=True)
        label_list = []
        for info in results:
            boxes = info.boxes
            for box in boxes:
                confidence = math.ceil(box.conf[0] * 100)
                if confidence >= 25:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = model.names[int(box.cls[0])]
                    print(label)
                    dic = {'Gender': label, 'Confidence': confidence}
                    label_list.append(dic)
        return label_list
    except Exception as e:
        print(f'>> Error in gender detection: {str(e)}')
        return []


def detect_GENDER (frame, cameraname):
    global x_gender
    gender_info = []
    path = ''
    try:
        # Assign image to model
        face_result = face_model.predict(frame)
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

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    path, gender_info = detect_GENDER(frame, 'E2')
    
    # Check for 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
