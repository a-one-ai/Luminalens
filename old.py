# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     form = SignUpForm()
#     if form.validate_on_submit():
#         users_collection = db['users']
#         new_user = {
#             'Username': form.username.data,
#             'Email': form.Email.data,
#             'Phone': form.phoneNumber.data,
#             'Password': form.password.data
#         }
#         users_collection.insert_one(new_user)
#         flash('Account created successfully!', 'success')
#         return redirect(url_for('signin'))
#     return render_template('signup.html', form=form)


# @app.route('/signin', methods=['GET', 'POST'])
# def signin():
#     form = SignInForm()
#     if form.validate_on_submit():
#         username = form.username.data
#         password = form.password.data
#         user = db['users'].find_one({'Username': username, 'Password': password})
#         if user:
#             user_obj = User()
#             user_obj.id = str(user['_id'])  
#             login_user(user_obj)
#             next_page = request.args.get('next')  
#             return redirect(next_page or url_for('apply_Model'))
#         else:
#             flash('Invalid username or password', 'error')
#     return render_template('signin.html', form=form)






# from pytube import YouTube
# from datetime import datetime
# from projectModel import *
# from pymongo import MongoClient
# from MongoPackageV2 import *
# import streamlink

# global capture
# def youtube(url):
#     try:
#         yt = YouTube(url)
#         stream = yt.streams.filter(res="720p", progressive=True).first()
#         if stream:
#             video_url = stream.url
#             return video_url
#         else:
#             print("No suitable stream found for the video.")
#             return None
#     except Exception as e:
#         print(f"Error in capturing YouTube video: {e}")
#         return None

# ##----------------------------------------
# def stream(url):
#     streams = streamlink.streams(url)
#     best_stream = streams["best"]
#     return best_stream.url
# ##----------------------------------------
# def readSource(srcType, src):
#     global capture
#     try:
#         if srcType == 'WEBCAM':
#             # src = int(src)
#             capture = cv2.VideoCapture(src , cv2.CAP_DSHOW)
#         elif srcType == 'RTSP':
#             # src = f'{src}'
#             capture = cv2.VideoCapture(src)
#         elif srcType == 'URL':
#             # src = f'{src}'
#             try:
#                 vsrc = youtube(src)
#                 capture = cv2.VideoCapture(vsrc)
#             except Exception as e:
#                 print(f"Error in capturing YouTube video: {e}")
#                 vsrc = stream(src)
#                 capture = cv2.VideoCapture(vsrc)
#     except Exception as e:
#         print(f"Error in readSource: {e}")
#         capture = None

#     return capture

# #_________________________________________________________
# def videoFeed(cameraName, modelName):
#     fps = 1
#     delay = int(1000 / fps)
#     modelName = f'{modelName}'
#     query = {'Camera Name': cameraName}

#     try :
#         src = int(find_existing_document(db['CameraInfo'],query)['Port'])
#     except :
#         src = str(find_existing_document(db['CameraInfo'],query)['Link'])

#     srcType = find_existing_document(db['CameraInfo'],query)['Source Type'] 

#     print(src , srcType)
#     cap = readSource(srcType, src)
    
#     if cap is None:
#         print("Error: Capture object is None.")
#         return
#     path = ''
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if modelName == 'violence':
#             path, res = violence(frame)
#         elif modelName == 'vehicle':
#             path, res = vehicleCounting(frame)
#         elif modelName == 'crowdedDensity':
#             path, res = crowdedDensity(frame)
#         elif modelName == 'crossingBorder':
#             path, res = crossingBorder(frame)
#         elif modelName == 'crowded':
#             path, res= crowded(frame)
#         elif modelName == 'Gender':
#             path , res = detect_GENDER(frame)

#         yield path, res, cameraName, modelName

#         if cv2.waitKey(delay) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# ##----------------------------------------
# def processInsert(cameraName, modelName):
    
#     generator = videoFeed(cameraName, modelName)

#     for result in generator:
#         path, res, cameraName, modelName, *extra_values = result

#         insert_model_info(cameraName, modelName, res ,path)




# #_________________________________________________________

# def run_models(modelName, frame, cameraName):
#     results = []

#     for model in modelName:
#         if model == 'violence':
#             path, res = violence(frame)
#             results.append((path, res, cameraName, model))
#         elif model == 'vehicle':
#             path, res = vehicleCounting(frame)
#             results.append((path, res, cameraName, model))
#         elif model == 'crowdedDensity':
#             path, res = crowdedDensity(frame)
#             results.append((path, res, cameraName, model))
#         elif model == 'crossingBorder':
#             path, res = crossingBorder(frame)
#             results.append((path, res, cameraName, model))
#         elif model == 'crowded':
#             path, res= crowded(frame)
#             results.append((path, res, cameraName, model))
#         elif model == 'Gender':
#             path , res = detect_GENDER(frame)
#             results.append((path, res, cameraName, model))

#     return results


# ##----------------------------------------
# def run_selected_models(selected_models, frame, cameraName):
#     all_results = []
#     for model in selected_models:
#         results = run_models([model], frame, cameraName)
#         all_results.extend(results)
#     return all_results


# ##----------------------------------------
# def videoFeedMulti(cameraName, modelNames):
#     fps = 1
#     delay = int(1000 / fps)
#     query = {'Camera Name': cameraName}

#     try:
#         src = int(find_existing_document(db['CameraInfo'], query)['Port'])
#     except:
#         src = str(find_existing_document(db['CameraInfo'], query)['Link'])

#     srcType = find_existing_document(db['CameraInfo'], query)['Source Type']

#     print(src, srcType)
#     cap = readSource(srcType, src)

#     if cap is None:
#         print("Error: Capture object is None.")
#         return


#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         all_results = run_selected_models(modelNames, frame, cameraName)
#         yield all_results
            
#     cap.release()
#     cv2.destroyAllWindows()

# ##----------------------------------------
# def multiModelRunInsert(cameraName, modelNames):
#     results_generator = videoFeedMulti(cameraName, modelNames)
#     for results in results_generator:
#         for result in results:
#             path, res, cameraName, modelName = result
#             insert_model_info(cameraName, modelName, res, path)









# # Define User class for Flask-Login
# class User(UserMixin):
#     pass

# # User loader function for Flask-Login
# @login_manager.user_loader
# def load_user(user_id):
#     user = User()
#     user.id = user_id
#     return user

# @app.before_request
# def before_request():
#     if not current_user.is_authenticated and request.endpoint and request.endpoint not in ['signin' , 'signup',
#                                                                                     'hello_world','add_camera', 'apply_Model' ,
#                                                                                         'get_cameraname',  'get_modelname', 'get_sourcetype', 'static']:
#         g.return_url = request.endpoint
#         return redirect(url_for('signin'))






# def videoFeed(cameraName, modelName):
#     """
#     Generates frames from a video feed and processes each frame using a specified model.

#     Args:
#         cameraName (str): Name of the camera.
#         modelName (str): Name of the model to process frames.

#     Yields:
#         tuple: (path, res, cameraName, modelName) - Path of the processed image, result from the model,
#             camera name, and model name.
#     """
#     query = {'Camera Name': cameraName} 

#     try : 
#         src = int(find_existing_document(db['CameraInfo'],query)['Port']) 
#     except : 
#         src = str(find_existing_document(db['CameraInfo'],query)['Link']) 
 
#     srcType = find_existing_document(db['CameraInfo'],query)['Source Type']  
 
#     print(src , srcType) 
     
#     cap , fps = readSource(srcType, src) 
     
#     if cap is None: 
#         print("Error: Capture object is None.") 
#         return 
 
#     count = 0 
#     while True: 
#         ret, frame = cap.read() 
#         if not ret: 
#             break 
         
#         count += 1  
#         if  (count % int(fps) == 0) or (count == 1):     
                
#             if modelName == 'violence': 
#                 path, res = violence(frame) 
#             elif modelName == 'vehicle': 
#                 path, res = vehicleCounting(frame) 
#             elif modelName == 'crowdedDensity': 
#                 path, res = crowdedDensity(frame) 
#             elif modelName == 'crossingBorder': 
#                 _, path, res = crossingBorder(frame) 
#             elif modelName == 'crowded': 
#                 path, res= crowded(frame) 
#             elif modelName == 'Gender': 
#                 path , res = detect_GENDER(frame) 

#             yield path, res, cameraName, modelName 

#         if cv2.waitKey(27) & 0xFF == ord('q'): 
#             break 
    
#     cap.release() 
#     cv2.destroyAllWindows() 

















# #Filter by Data and Get Average of Count in Form of Time Range
# def date_filter_aggerigates_html(CameraName, ModelName,TargetDate) :
    
#     # Determine the appropriate collection based on the ModelName
#     if ModelName == 'violence':
#         existing_collection = db['ModelViolenceData']
#     elif ModelName == 'vehicle':
#         existing_collection = db['ModelVehicleData']
#     elif ModelName == 'crowdedDensity':
#         existing_collection = db['ModelDensityData']
#     elif ModelName == 'crossingBorder':
#         existing_collection = db['ModelCountingData']
#     elif ModelName == 'crowded':
#         existing_collection = db['ModelCrowdedData']
        
#     query = {'Camera Info.Camera Name': CameraName}
    
#     if check_existing_document(existing_collection, query):
#         print(f'Camera Found in {ModelName} Collection')
#         # Create the aggregation pipeline
#         pipeline = [
#             {
#                 "$match": {"Date": TargetDate}
#             },
#             {
#                 "$group": {
#                     "_id": {"$hour": "$Timestamp"},
#                     "count": {"$avg": "$Count"}
#                 }
#             },
#             {
#                 "$project": {
#                     "Hour": "$_id",
#                     "Count Average": "$count",
#                     "_id": 0
#                 }
#             },
#             {
#                 "$sort": {"Hour": 1}
#             }
#         ]                                                

#         result = list(existing_collection.aggregate(pipeline))  
#         # Generate HTML table directly with spaces and formatting
#         html_table = (
#             "<table border='1'>"
#             "<tr><th style='text-align:left;'>Time Range</th>"
#             "<th style='text-align:center;'>Count Average</th></tr>"
#         )

#         for item in result:
#             hour = item['Hour']
#             average_count = math.ceil(item['Count Average'])

#             # Determine AM or PM based on the hour
#             am_pm = "PM" if (hour+2) >= 12 else "AM"
#             formatted_hour = (hour+2) if (hour+2) <= 12 else (hour+2) - 12

#             # Format time range
#             time_range = f"{formatted_hour} {am_pm} - {formatted_hour + 1} {am_pm}"

#             # Add a row to the HTML table
#             html_table += (
#                 f"<tr><td style='text-align:left;'>{time_range}</td>"
#                 f"<td style='text-align:center;'>{average_count}</td></tr>"
#             )

#         # Close the HTML table
#         html_table += "</table>"

#         return html_table    
#     else :
#         return f'Camera not Found in {ModelName} Collection'
                       
#TargetDate = '2024-02-01'              
#aggerigates = date_filter_aggerigates_html('Density_Cam','crowdedDensity',TargetDate)
#print(aggerigates)



# #Filter by Data and Get Average of Count in Form of Time Range
# def date_filter_aggerigates_df(CameraName, ModelName,day , month,year) :

#     if int(month) < 10:
#         month = '0' + str(month)
#     if int(day) < 10:
#         day = '0' + str(day)

#     TargetDate = str(year) + '-' + str(month) + '-' + str(day)    

#     # Determine the appropriate collection based on the ModelName
#     if ModelName == 'violence':
#         existing_collection = db['ModelViolenceData']
#     elif ModelName == 'vehicle':
#         existing_collection = db['ModelVehicleData']
#     elif ModelName == 'crowdedDensity':
#         existing_collection = db['ModelDensityData']
#     elif ModelName == 'crossingBorder':
#         existing_collection = db['ModelCountingData']
#     elif ModelName == 'crowded':
#         existing_collection = db['ModelCrowdedData']
        
#     query = {'Camera Info.Camera Name': CameraName}
    
#     if check_existing_document(existing_collection, query):
#         print(f'{CameraName} Camera Found in {ModelName} Collection')
#         # Create the aggregation pipeline
#         pipeline = [
#             {
#                 "$match": {"Date": TargetDate}
#             },
#             {
#                 "$group": {
#                     "_id": {"$hour": "$Timestamp"},
#                     "count": {"$avg": "$Count"}
#                 }
#             },
#             {
#                 "$project": {
#                     "Hour": "$_id",
#                     "Count Average": "$count",
#                     "_id": 0
#                 }
#             },
#             {
#                 "$sort": {"Hour": 1}
#             }
#         ]                                                

#         result = list(existing_collection.aggregate(pipeline))  
#         # Generate HTML table directly with spaces and formatting
#     if result:
        
#         data = []

#         for item in result:
#             hour = item['Hour']
#             average_count = math.ceil(item['Count Average'])

#             # Determine AM or PM based on the hour
#             am_pm = "PM" if (hour + 2) >= 12 else "AM"
#             formatted_hour = (hour + 2) if (hour + 2) <= 12 else (hour + 2) - 12

#             # Format time range
#             time_range = f"{formatted_hour} {am_pm} - {formatted_hour + 1} {am_pm}"

#             data.append({'Time Range': time_range, 'Count Average': average_count})
            
#         #Get Pandas DataFrame
#         df = pd.DataFrame(data)
#         return df

#     else:
#         return f'{CameraName} Camera not Found in {ModelName} Collection'



#________________________________________________________________
# #Importing Packages
# from pymongo import MongoClient
# from datetime import datetime
# import pandas as pd
# import pytz
# import math
# from pymongo.errors import OperationFailure
# from time import sleep

# #Assign Client Connection and Database as Global
# global client , db


# #Connect to the default local MongoDB instance
# client = MongoClient('mongodb://localhost:27017/')
# #Connect to Databse
# db = client.CameraProject2


# #Checking Existing Documents 
# def check_existing_document(existing_collection, query):
#     return existing_collection.find_one(query) is not None


# #Finding Existing Documents 
# def find_existing_document(existing_collection, query):
#     return existing_collection.find_one(query)


# #Updating  Existing Documents 
# def update_existing_document(existing_collection, query, update_data):
#     # Update an existing document based on the query with the provided update_data
#     result = existing_collection.update_one(query, {'$set': update_data})
#     return result.modified_count

# #________________________________________________________


# #Insert Camera Information
# def insert_camera_info(CameraName, SourceType, Source):
    
#     # Connect to collection
#     existing_collection = db['CameraInfo']

#     # Get current UTC time
#     current_time_utc = datetime.utcnow()

#     # Convert UTC time to Egypt timezone
#     egypt_tz = pytz.timezone('Africa/Cairo')
#     current_time_egypt = current_time_utc.replace(tzinfo=pytz.utc).astimezone(egypt_tz)

#     # Format the current time in YYYY-MM-DD format
#     date_str = current_time_egypt.strftime("%Y-%m-%d")

#     # Prepare data for insertion
#     data = {
#         'Camera Name': CameraName,
#         'Source Type': SourceType,
#     }

#     # Add specific fields based on the SourceType
#     if SourceType == 'WEBCAM':
#         data['Port'] = int(Source)
#     elif SourceType in ['RTSP', 'URL']:
#         data['Link'] = Source

#     # Add location coordinates, status, and timestamp information
#     data['Status'] = 'OFF'
#     data['Insertion Timestamp'] = current_time_egypt  # Store the timestamp as a datetime object
#     data['Insertion Date'] = date_str

#     # Check if the document with the given Camera Name already exists
#     query = {'Camera Name': CameraName}
#     if check_existing_document(existing_collection, query):
#         print('This Camera Name Already Exists')
#     else:
#         # Insert the document into the collection
#         inserted_document = existing_collection.insert_one(data)
#         # Print a success message with the inserted document ID
#         print('Inserted Successfully with ID:', inserted_document.inserted_id)
#         # Return the inserted document (optional, depending on your needs)
#         return inserted_document





# #_________________________________________________________________



# #Inserting Model Information
# def insert_model_info(CameraName, ModelName, Label, FramePath):
    
#     # Determine the appropriate collection based on the ModelName
#     if ModelName == 'violence':
#         existing_collection = db['ModelViolenceData']
#     elif ModelName == 'vehicle':
#         existing_collection = db['ModelVehicleData']
#     elif ModelName == 'crowdedDensity':
#         existing_collection = db['ModelDensityData']
#     elif ModelName == 'crossingBorder':
#         existing_collection = db['ModelCountingData']
#     elif ModelName == 'crowded':
#         existing_collection = db['ModelCrowdedData']
#     elif ModelName == 'Gender':
#         existing_collection = db['ModelGenderData']

#     # Get the current date and time in UTC
#     current_time_utc = datetime.utcnow()

#     # Define the timezone for Egypt (Eastern European Time)
#     egypt_tz = pytz.timezone('Africa/Cairo')

#     # Convert UTC time to Egypt timezone
#     current_time_egypt = current_time_utc.replace(tzinfo=pytz.utc).astimezone(egypt_tz)


#     # Extract the date component
#     date_str = current_time_egypt.strftime("%Y-%m-%d")

#     # Prepare data for insertion
#     data = {
#         'Model Name': ModelName,
#     }

#     # Check if the camera with the given name exists
#     query = {'Camera Name': CameraName}
#     camera_collection = db['CameraInfo']
#     if check_existing_document(camera_collection, query):
#         print('Camera Found')
        
#         # Update the camera status to 'ON'
#         update_existing_document(camera_collection, query, {'Status': 'ON'})
        
#         # Retrieve camera data
#         camera_data = find_existing_document(camera_collection, query)
        
#         # Add camera information to the data
#         data['Camera Info'] = camera_data

#     else:
#         print('Camera Not Added in Camera Collection')
#         return 'Camera Not Added in Camera Collection'

#     # Check the type of Label and set Count or Label accordingly
#     if isinstance(Label, int) or ModelName in ['vehicle', 'crowdedDensity', 'crossingBorder', 'crowded']:
#         data['Count'] = Label
#     elif isinstance(Label, str) or ModelName not in ['vehicle', 'crowdedDensity', 'crossingBorder', 'crowded']:
#         data['Label'] = Label

#     # Add Frame Path, Timestamp, and Date information
#     data['Frame Path'] = FramePath
#     data['Timestamp'] = current_time_egypt
#     data['Date'] = date_str

#     # Insert the document into the collection
#     inserted_document = existing_collection.insert_one(data)
    
#     # Print a success message with the inserted document ID
#     print(f'Inserted Successfully with ID in {ModelName} Collection: {inserted_document.inserted_id}')
#     return inserted_document

# #insert_model_info('Density_Cam','crowdedDensity',80,'352.png')



# #_________________________________________________________________




# #Returning all camera names in DB 
# def finding_camera_names():
#     db = client.CameraProject2    
#     existing_collection = db['CameraInfo']
#     cursor = existing_collection.find({})
#     camera_names = [document['Camera Name'] for document in cursor]
#     return camera_names


# #_______________________________________________________


# #Filter by Data and Get Average of Count in Form of Time Range
# def date_filter_aggerigates_df(CameraName, ModelName, day, month, year):
#     if int(month) < 10:
#         month = '0' + str(month)
#     if int(day) < 10:
#         day = '0' + str(day)

#     TargetDate = str(year) + '-' + str(month) + '-' + str(day)    

#     # Determine the appropriate collection based on the ModelName
#     if ModelName == 'violence':
#         existing_collection = db['ModelViolenceData']
#     elif ModelName == 'vehicle':
#         existing_collection = db['ModelVehicleData']
#     elif ModelName == 'crowdedDensity':
#         existing_collection = db['ModelDensityData']
#     elif ModelName == 'crossingBorder':
#         existing_collection = db['ModelCountingData']
#     elif ModelName == 'crowded':
#         existing_collection = db['ModelCrowdedData']
#     elif ModelName == 'Gender':
#         existing_collection = db['ModelGenderData']
        
#     query = {'Camera Info.Camera Name': CameraName}
    
#     if check_existing_document(existing_collection, query):
#         print(f'{CameraName} Camera Found in {ModelName} Collection')
#         # Create the aggregation pipeline
#         pipeline = [
#             {
#                 "$match": {"Date": TargetDate}
#             },
#             {
#                 "$group": {
#                     "_id": {"$hour": "$Timestamp"},
#                     "count": {"$avg": "$Count"}
#                 }
#             },
#             {
#                 "$project": {
#                     "Hour": "$_id",
#                     "Count Average": "$count",
#                     "_id": 0
#                 }
#             },
#             {
#                 "$sort": {"Hour": 1}
#             }
#         ]                                                

#         result = list(existing_collection.aggregate(pipeline))  
        
#         if result:
#             data = []

#             for item in result:
#                 hour = item['Hour']
#                 average_count = math.ceil(item['Count Average'])

#                 # Determine AM or PM based on the hour
#                 am_pm = "PM" if (hour + 2) >= 12 else "AM"
#                 formatted_hour = (hour + 2) if (hour + 2) <= 12 else (hour + 2) - 12

#                 # Format time range
#                 time_range = f"{formatted_hour} {am_pm} - {formatted_hour + 1} {am_pm}"

#                 data.append({'Time Range': time_range, 'Count Average': average_count})

#             #Get Pandas DataFrame
#             df = pd.DataFrame(data)
#             return df
#         else:
#             return pd.DataFrame()  # Return an empty DataFrame if no data found
#     else:
#         return pd.DataFrame()  # Return an empty DataFrame if camera not found




# #_________________________________________________________________
# #Filter by Data and Get Average of Count in Form of Time Range 
# def average_camera_count(CameraName, ModelName): 
 
 
#     # Determine the appropriate collection based on the ModelName 
#     if ModelName == 'violence': 
#         existing_collection = db['ModelViolenceData'] 
#     elif ModelName == 'vehicle': 
#         existing_collection = db['ModelVehicleData'] 
#     elif ModelName == 'crowdedDensity': 
#         existing_collection = db['ModelDensityData'] 
#     elif ModelName == 'crossingBorder': 
#         existing_collection = db['ModelCountingData'] 
#     elif ModelName == 'crowded': 
#         existing_collection = db['ModelCrowdedData'] 
#     elif ModelName == 'Gender': 
#         existing_collection = db['ModelGenderData'] 
         
#     query = {'Camera Info.Camera Name': CameraName} 
     
#     if check_existing_document(existing_collection, query): 
#         print(f'{CameraName} Camera Found in {ModelName} Collection') 
#         pipeline = [ 
#             { 
#                 "$match": query 
#             }, 
#             { 
#                 "$group": { 
#                     "_id": "$Camera Info.Camera Name", 
#                     "count": {"$avg": "$Count"} 
#                 } 
#             }, 
#             { 
#                 "$project": { 
#                     "Camera Name": "$_id", 
#                     "Count Average": "$count", 
#                     "_id": 0 
#                 } 
#             }, 
#             { 
#                 "$sort": {"Camera Name": 1} 
#             } 
#         ] 
 
#         try : 
#             result = list(existing_collection.aggregate(pipeline))   
             
#             if result: 
#                 result = result[0] 
#                 result['Count Average'] = math.ceil(result['Count Average']) 
#                 result['Model'] = ModelName             
#                 #df = pd.DataFrame(result) 
#                 #print(result[0]) 
#                 #return df 
#                 print(result) 
                 
#                 return result 
             
#         except :     
#                 nulldic = {'Camera Name' : CameraName , 
#                             'Model' : ModelName , 
#                             'Count Average' : 'Cannot Calculate' 
#                     } 
                 
#                 return nulldic 
             
 
#     else: 
#         nulldic = {'Camera Name' : CameraName , 
#                     'Model' : ModelName , 
#                     'Count Average' : 'Not Available Data' 
#                     }         
         
#         return  nulldic 
 
# #print(average_camera_count('Elsisi','crossingBorder'))















# from modelsReq.violence.model import Model
# import cv2
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from modelsReq.density.src.models.CSRNet import CSRNet
# from datetime import datetime
# from ultralytics import YOLO
# from modelsReq.yoloModels.tracker import Tracker
# import pandas as pd
# import math
# import threading


# #-----------initialize for models------------
# modelCrowd  = None
# model= None
# modelV  = None
# modelDEns = None
# modelG = None

# model_lock = threading.Lock()
# count = 0
# def initialize_models():
#     global count 
#     count += 1
#     print("Initializing models for the {} time".format(count))
#     global modelCrowd, model , modelV , modelDEns , modelG
#     with model_lock:
#         if modelCrowd is None:
#             modelCrowd = YOLO('app/modelsReq/yoloModels/best_crowded.pt')

#         if model is None :
#              model = YOLO('app/modelsReq/yoloModels/yolov8s.pt')

#         if modelV is None:
#              modelV = Model()

#         if modelDEns is None:
#              modelDEns = CSRNet()

#         if modelG is None:
#              modelG = YOLO('app/modelsReq/yoloModels/gender.pt')


        
# initialize_models()


# my_file = open("app/modelsReq/yoloModels/coco.txt", "r")
# data = my_file.read()
# class_list = data.split("\n")
# tracker = Tracker()




# #-----------gateCounting model -------------


# #-----------density models -------------

# modelDEns = CSRNet()
# PATH = 'https://huggingface.co/muasifk/CSRNet/resolve/main/CSRNet.pth'
# state_dict = torch.hub.load_state_dict_from_url(PATH, map_location=torch.device('cpu'))
# modelDEns.load_state_dict(state_dict)
# modelDEns.eval()
# print('\n Model loaded successfully.. \n')

# global x_density
# x_density = 0 

# def crowdedDensity(frame):
#     global x_density

#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame = frame.astype(np.float32) / 255  # normalize image
#     frame = torch.from_numpy(frame).permute(2, 0, 1)  # reshape to [c, w, h]

#     # predict
#     predict = modelDEns(frame.unsqueeze(0))
#     count = predict.sum().item()

#     # Plot the results using Matplotlib
#     fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 3))
#     ax0.imshow(frame.permute(1, 2, 0))  # reshape back to [w, h, c]
#     ax1.imshow(predict.squeeze().detach().numpy(), cmap='jet')
#     ax0.set_title('People Count')
#     ax1.set_title(f'People Count = {count:.0f}')
#     ax0.axis("off")
#     ax1.axis("off")
#     plt.tight_layout()

#     # Save the figure
#     x_density = x_density + 1
#     path = f'app/output/density/figure{x_density}.jpg'
#     plt.savefig(path)  # Specify the desired output file path and format
#     plt.close()  # Close the figure to release resources
#     print('Figure saved successfully.')

#     return path, count


# #-----------crowded model -------------
# global x_crowd
# x_crowd = 0
# def crowded(frame):
#     global x_crowd
#     count = 0
#     results = modelCrowd(frame,stream=True)

#         # Getting bbox,confidence and class names informations to work with
#         # Assign image to model to detect people and get boxes
#     for info in results:
#             boxes = info.boxes
#             for box in boxes:
#                 confidence = box.conf[0]
#                 confidence = math.ceil(confidence * 100)
#                 Class = int(box.cls[0])     
#                 # Add box if confidence of detection more than or eqaul to 30% and count objects
#                 if confidence >= 40:
#                     x1,y1,x2,y2 = box.xyxy[0]
#                     x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
#                     cv2.rectangle(frame,(x1,y1),(x2,y2),(0, 255, 0),2)
#                     count +=1
                    
#     cv2.putText(frame, f"Count : {count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 5)
#     x_crowd = x_crowd + 1
#     path = f'app/output/crowded/figure{x_crowd}.jpg'
#     cv2.imwrite(path , frame)
 
#     return path ,  count




# #-----------crossingBorder model -------------
# global x_crossing
# x_crossing = 0

# def crossingBorder(frame):
#     global x_crossing
#     count = 0  
#     results = model.predict(frame)
#     # Print the type and content of 'results' for debugging
#     print(f"Type of 'results': {type(results)}")
#     print(f"Content of 'results': {results}")
#     a = results[0].boxes.data
#     px = pd.DataFrame(a).astype("float")

#     bbox_list = []
#     for index, row in px.iterrows():
#         x1 = int(row[0])
#         y1 = int(row[1])
#         x2 = int(row[2])
#         y2 = int(row[3])
#         d = int(row[5])
#         c = class_list[d]
#         if 'person' in c:
#             bbox_list.append([x1, y1, x2, y2])
#             count += 1  

#     bbox_id = tracker.update(bbox_list)
#     for bbox in bbox_id:
#         x3, y3, x4, y4, d = bbox
#         cx = (x3 + x4) // 2
#         cy = (y3 + y4) // 2
#         cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
#         cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)

#     cv2.putText(frame, f'Count: {count}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#     # cv2.imshow('dd' , frame)
#     x_crossing += 1
#     path = f'app/output/crossing/figure{x_crossing}.jpg'
#     cv2.imwrite(path, frame)
#     return path, count



# #-----------vehicleCounting models -------------
# x_vehicle = 0
# def vehicleCounting(frame):
#     count = 0  

#     results = model.predict(frame)
#     a = results[0].boxes.data
#     px = pd.DataFrame(a).astype("float")

#     l = []
#     for index, row in px.iterrows():
#         x1 = int(row[0])
#         y1 = int(row[1])
#         x2 = int(row[2])
#         y2 = int(row[3])
#         d = int(row[5])
#         c = class_list[d]
#         # Detect vehicles 
#         if 'car' in c or 'truck' in c or 'bus' in c or 'bicycle' in c or 'motorcycle' in c:
#             list.append([x1, y1, x2, y2])
#             count += 1  

#     bbox_id = tracker.update(list)
#     for bbox in bbox_id:
#         x3, y3, x4, y4, d = bbox
#         cx = (x3 + x4) // 2
#         cy = (y3 + y4) // 2
#         # cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
#         cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)

#     cv2.putText(frame, f'Count: {count}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#     x_vehicle = x_vehicle + 1
#     path = f'app/output/vehicle/figure{x_vehicle}.jpg'
#     cv2.imwrite(path , frame)
#     return path , count






# #-----------violence model --------------
# global x_violence
# x_violence = 0


# def violence(frame):
#     global x_violence
#     RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     predictions = modelV.predict(image=RGBframe)
#     label = predictions['label']
#     if label in ['violence in office', 'fight on a street','street violence'] :
#                 label = 'Predicted Violence'
#     cv2.putText(frame, f'This is a {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#     x_violence = x_violence + 1
#     path = ''
#     if label == 'Predicted Violence':
#         path = f'app/output/violence/figure{x_violence}.jpg'
#         cv2.imwrite(path , frame)

#     return path , label 



# #-----------Accidents model --------------



# #-----------Gender model ------------------
# global x_gender
# x_gender = 0
# def detect_GENDER(frame):
#     global x_gender
#     try:
#         # Assign image to model
#         results = modelG(frame, stream=True)

#         # Getting bbox, confidence, and class names information to work with
#         # Assign image to model to detect people and get boxes
#         for info in results:
#             boxes = info.boxes
#             for box in boxes:
#                 confidence = box.conf[0]
#                 confidence = math.ceil(confidence * 100)
#                 Class = int(box.cls[0])
#                 # Add box if confidence of detection more than or equal to 40% and count objects
#                 if confidence >= 40 :
#                     x1, y1, x2, y2 = box.xyxy[0]
#                     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 4)
#                     # Display label
#                     label = modelG.names[int(Class)]
#                     font = cv2.FONT_HERSHEY_SIMPLEX
#                     font_scale = 0.8
#                     font_thickness = 2
#                     text_color = (0, 120, 255)
#                     cv2.putText(frame, f"{label}: {confidence}%", (x1, y1 - 10),
#                                 font, font_scale, text_color, font_thickness)
                    
#                     x_gender = x_gender + 1
#                     path = ''
#                     path = f'app/output/gender/figure{x_gender}.jpg'
#                     cv2.imwrite(path , frame)
                    
#         return path , label

#     except Exception as e:
#         print(f'>> Error: {str(e)}')
#         return None , None




















