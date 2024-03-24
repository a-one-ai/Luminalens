from pytube import YouTube 
import streamlink   
import cv2  
from threading import Thread
from projectModel import *  
from MongoPackageV2 import *
from MongoPackageV2 import update_camera_status_models_collection

import time  
import warnings
warnings.filterwarnings("ignore")



stop_processing = False  # Global flag to control processing
# ##--------------------------------------------------------

def stop_processing_function():
    global stop_processing
    stop_processing = True

# ##--------------------------------------------------------
def reset_processing_flag():
    global stop_processing
    stop_processing = False

# Function to stop processing for a specific camera
camera_stop_flags = {}    
# ##--------------------------------------------------------
def stop_processing_for_camera(camera_name):
    global camera_stop_flags
    camera_stop_flags[camera_name] = True

# Function to reset processing for a specific camera
# ##--------------------------------------------------------    
def reset_processing_for_camera(camera_name):
    global camera_stop_flags
    camera_stop_flags[camera_name] = False    

# Define model_stop_flags as a global variable
model_stop_flags = {}

# Function to enable processing for a specific model on a specific camera
# ##--------------------------------------------------------
def enable_model_for_camera(camera_name, model_name):
    global model_stop_flags
    model_stop_flags[(camera_name, model_name)] = False

# Function to disable processing for a specific model on a specific camera
# ##--------------------------------------------------------    
def disable_model_for_camera(camera_name, model_name):
    global model_stop_flags
    model_stop_flags[(camera_name, model_name)] = True

def youtube(url):
    """
    Extracts the URL of the best available stream from a YouTube video URL.

    Args:
        url (str): YouTube video URL.

    Returns:
        str: URL of the best available stream, or None if no suitable stream is found.
    """
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(res="720p", progressive=True).first()
        if stream:
            video_url = stream.url
            return video_url
        else:
            print("No suitable stream found for the video.")
            return None
    except Exception as e:
        print(f"Error in capturing YouTube video: {e}")
        return None

# ##--------------------------------------------------------
def stream(url):
    """
    Extracts the URL of the best available stream from a given URL using Streamlink.

    Args:
        url (str): URL of the video stream.

    Returns:
        str: URL of the best available stream.
    """
    streams = streamlink.streams(url)
    best_stream = streams["best"]
    return best_stream.url

# ##--------------------------------------------------------
def readSource(srcType, src):
    """
    Reads the video source and initializes a video capture object.

    Args:
        srcType (str): Type of the video source ('WEBCAM', 'RTSP', or 'URL').
        src (str): Source identifier (e.g., port number for 'WEBCAM' and 'RTSP', URL for 'URL').

    Returns:
        cv2.VideoCapture: Video capture object.
    """
    capture = ''
    try: 
        if srcType == 'WEBCAM': 
            capture = cv2.VideoCapture(int(src)) 
            fps = 30 

        elif srcType == 'RTSP': 
            capture = cv2.VideoCapture(src) 
            if 'rtsp' in src : 
                fps = 30 

            else : 
                fps = capture.get(cv2.CAP_PROP_FPS)                 

        elif srcType == 'URL': 
            try: 
                vsrc = youtube(src) 
                if vsrc is not None :
                    print('Reading Youtube Video Using Pytube')
                    capture = cv2.VideoCapture(vsrc) 
                    fps = capture.get(cv2.CAP_PROP_FPS) if capture.isOpened() else 30    
                else : 
                    print('Reading Youtube Live Using Streamlink')
                    vsrc = stream(src) 
                    capture = cv2.VideoCapture(vsrc) 
                    fps = capture.get(cv2.CAP_PROP_FPS) if capture.isOpened() else 30  # Default to 30 fps if not available    
                
            except Exception as e: 
                print(f"Error in capturing entered video: {e}") 
                

        return capture , fps     
                            
    except Exception as e: 
        print(f"Error in readSource: {e}") 
        capture = None 
        return capture 

# ##--------------------------------------------------------
def process_frame(frame, modelName, cameraName):
    """
    Process a single frame using the specified model.

    Args:
        frame (numpy.ndarray): Input frame.
        modelName (str): Name of the model to process the frame.

    Returns:
        tuple: (path, res) - Path of the processed image and result from the model.
    """
    try:
        if modelName == 'violence':
            path, res = violence(frame)
            return  path, res 

        elif modelName == 'vehicle':
            path, res = vehicleCounting(frame)
            return path, res 

        elif modelName == 'crowdedDensity':
            path, res = crowdedDensity(frame)
            return path, res 

        elif modelName == 'crossingBorder':
            path, res = crossingBorder(frame, cameraName)
            return path, res

        elif modelName == 'crowded':
            path, res = crowded(frame)
            return path, res 

        elif modelName == 'gender':
            path, res = detect_GENDER(frame, cameraName)
            return path, res
        
        elif modelName == 'Age':
            path, res = Age_detection(frame, cameraName)
            return path, res
    except Exception as e:
        print(f"Error occurred while processing frame with model '{modelName}': {e}")
        return None, None


# ##--------------------------------------------------------
def videoFeed(cameraName, modelName):
    """
    Generates frames from a video feed and processes each frame using a specified model.

    Args:
        cameraName (str): Name of the camera.
        modelName (str): Name of the model to process frames.

    Yields:
        tuple: (path, res, cameraName, modelName) - Path of the processed image, result from the model,
            camera name, and model name.
    """
    query = {'Camera Name': cameraName}

    try:
        src = int(find_existing_document(db['CameraInfo'], query)['Port'])
    except  Exception:
        src = str(find_existing_document(db['CameraInfo'], query)['Link'])

    srcType = find_existing_document(db['CameraInfo'], query)['Source Type']

    print(src, srcType)

    cap, fps = readSource(srcType, src)

    if cap is None:
        print("Error: Capture object is None.")
        return

    count = 0
    path = ''
    res = ''
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % int(fps) == 0 or count == 1:
            result = process_frame(frame, modelName, cameraName)
            if result is not None:
                path, res = result
            yield path, res, cameraName, modelName

        if cv2.waitKey(27) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ##--------------------------------------------------------
def processInsert(cameraName, modelName):
    """
    Processes video frames from a camera for a specified model and inserts the results into a database.

    Args:
        cameraName (str): Name of the camera.
        modelName (str): Name of the model to process frames.
    """
    generator = videoFeed(cameraName, modelName)

    for result in generator:
        path, res, cameraName, modelName, *extra_values = result

        insert_model_info(cameraName, modelName, res, path)

# #_________________________________________________________
# Modify the run_models function
def run_models(model_names, frame, camera_name):
    global model_stop_flags  # Make sure to use the global variable
    results = []

    try:
        for model_name in model_names:
            if model_stop_flags.get((camera_name, model_name), False):
                print(f"Model '{model_name}' is disabled for camera '{camera_name}'.")
                continue

            if model_name == 'violence':
                path, res = violence(frame, camera_name)
                results.append((path, res, camera_name, model_name))

            elif model_name == 'vehicle':
                path, res = vehicleCounting(frame, camera_name)
                results.append((path, res, camera_name, model_name))

            elif model_name == 'crowdedDensity':
                path, res = crowdedDensity(frame, camera_name)
                results.append((path, res, camera_name, model_name))

            elif model_name == 'crossingBorder':
                path, res = crossingBorder(frame, camera_name)
                results.append((path, res, camera_name, model_name))

            elif model_name == 'crowded':
                path, res = crowded(frame, camera_name)
                results.append((path, res, camera_name, model_name))

            elif model_name == 'gender':
                path, res = detect_GENDER(frame, camera_name)
                results.append((path, res, camera_name, model_name))

            elif model_name == 'clothes color':
                path, res = human_clothing(frame, camera_name)
                results.append((path, res, camera_name, model_name))
                
            elif model_name == 'Enter Exit Counting':
                path, res = enterExitCounting(frame, camera_name)
                results.append((path, res, camera_name, model_name))

            elif model_name == 'Age':
                path, res = Age_detection(frame, camera_name)
                results.append((path, res, camera_name, model_name))

    except Exception as e:
        print(f"Error occurred while running models: {e}")
    return results

# ##--------------------------------------------------------
def run_selected_models(selected_models, frame, cameraName):
    """
    Runs selected models on a single frame obtained from a video feed.

    Args:
        selected_models (list of str): List of selected model names to process frames.
        frame (numpy.ndarray): Single frame obtained from the video feed.
        cameraName (str): Name of the camera.

    Returns:
        list: List of tuples containing the path of the processed image, result from each selected model,
            camera name, and model name.
    """
    all_results = []

    try:
        for model in selected_models:
            results = run_models([model], frame, cameraName)
            all_results.extend(results)
    except Exception as e:
        print(f"Error occurred while running selected models: {e}")
        return []

    return all_results


# ##--------------------------------------------------------
def videoFeedMulti(cameraName, modelNames):
    """
    Generates frames from a video feed and processes each frame using multiple specified models.

    Args:
        cameraName (str): Name of the camera.
        modelNames (list of str): List of model names to process frames.

    Yields:
        list: List of tuples containing the path of the processed image, result from each model,
            camera name, and model name for each frame.
    """

    query = {'Camera Name': cameraName}

    try:
        src = int(find_existing_document(db['CameraInfo'], query)['Port'])
    except Exception:
        src = str(find_existing_document(db['CameraInfo'], query)['Link'])

    srcType = find_existing_document(db['CameraInfo'], query)['Source Type']

    print(src, srcType)
    cap , fps = readSource(srcType, src)

    if cap is None:
        print("Error: Capture object is None.")
        return
    
    count = 0 
    try:
    
        while True:
            ret, frame = cap.read()
            if not ret or stop_processing  or  camera_stop_flags.get(cameraName) :
                print('Stop Processing : ',stop_processing)
                print('Stop Processing For This Cam : ',camera_stop_flags.get(cameraName))
                query = {'Camera Name': cameraName}
                camera_collection = db['CameraInfo']
                if check_existing_document(camera_collection, query):
                    update_existing_document(camera_collection, query, {'Status': 'OFF'}) 
                    print('Status is OFF')     
                    update_camera_status_models_collection(cameraName = cameraName)                       
                break

            count += 1  
            if  (count % int(fps) == 0) or (count == 1):     
                    print('Stop Processing : ',stop_processing)
                    all_results = run_selected_models(modelNames, frame, cameraName)
                    yield all_results

            if stop_processing:
                break         
            if cv2.waitKey(27) & 0xFF == ord('q'): 
                break 
    except Exception as e:
        print(f"Error occurred during video processing: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ##----------------------------------------
# def multiModelRunInsert(cameraName, modelNames):
#     """
#     Processes video frames from a camera for multiple specified models simultaneously and inserts the results into a database.

#     Args:
#         cameraName (str): Name of the camera.
#         modelNames (list of str): List of model names to process frames.
#     """
#     try:
#         results_generator = videoFeedMulti(cameraName, modelNames)
#         for results in results_generator:
#             for result in results:
#                 path, res, cameraName, modelName = result
#                 insert_model_info(cameraName, modelName, res, path)
#     except Exception as e:
#         print(f"Error in multiModelRunInsert: {e}")
#     finally:
#         # Release resources
#         cv2.destroyAllWindows()

# ##--------------------------------------------------------
def multiModelRunInsert(cameraName, modelNames):
    """
    Processes video frames from a camera for multiple specified models simultaneously and inserts the results into a database.

    Args:
        cameraName (str): Name of the camera.
        modelNames (list of str): List of model names to process frames.
    """
    path = ""
    try:
        reset_processing_flag()        
        reset_processing_for_camera(cameraName)        
        results_generator = videoFeedMulti(cameraName, modelNames)
        for results in results_generator:
            for result in results:
                path, res, cameraName, modelName = result
                print(res)
                # Convert res to a compatible type
                res_encoded = res.tolist() if isinstance(res, np.ndarray) else res
                try:
                    if  (isinstance (res_encoded,list) ) and (len(res_encoded) == 0):
                        continue                    
                    status = insert_model_info(cameraName, modelName, res_encoded, path)
                    print(status)
                    status = status['Inserted']
                except Exception as insert_error:
                    print(f"Error inserting model info for path '{path}': {insert_error}")
    except Exception as e:
        print(f"Error in multiModelRunInsert: {e}")
    finally:
        # Release resources
        cv2.destroyAllWindows()


  
satus_list = []

def multiModelRunInsert_Testing(cameraName, modelNames, duration=10):
    """
    Processes video frames from a camera for multiple specified models simultaneously and inserts the results into a database.

    Args:
        cameraName (str): Name of the camera.
        modelNames (list of str): List of model names to process frames.
        duration (int): Duration in seconds for running the function.
    """
    start_time = time.time()
    global satus_list
    try:
        reset_processing_flag()        
        reset_processing_for_camera(cameraName)        
        results_generator = videoFeedMulti(cameraName, modelNames)
        for results in results_generator:
            for result in results:
                path, res, cameraName, modelName = result
                print(res)
                # Convert res to a compatible type
                res_encoded = res.tolist() if isinstance(res, np.ndarray) else res
                try:
                    if  (isinstance (res_encoded,list) ) and (len(res_encoded) == 0):
                        continue                    
                    status = insert_model_info(cameraName, modelName, res_encoded, path)
                    satus_list.append(status)
                    
                except Exception as insert_error:
                    print(f"Error inserting model info for path '{path}': {insert_error}")
                    satus_list.append(False)
                # Check if the elapsed time exceeds the specified duration
                if time.time() - start_time >= duration:
                    return
                
    except Exception as e:
        print(f"Error in multiModelRunInsert: {e}")
        satus_list.append(False)

    finally:
        # Release resources
        cv2.destroyAllWindows()

def apply_Model_Testing(cameraName, modelNames, duration=10):
    # Start the multiModelRunInsert function in a separate thread
    task_thread = Thread(target=multiModelRunInsert_Testing, args=(cameraName, modelNames, duration))
    task_thread.start()
    print("Testing multiModelRunInsert...")
    time.sleep(duration)
    
    # If the function is still running, stop it
    if task_thread.is_alive():
    # Block on the thread until it completes its execution
        print("Stopping multiModelRunInsert...")
        task_thread.join()     

    print('Testing Completed')
            

# # # # Example usage:
# apply_Model_Testing(cameraName="aslom", modelNames=["crowded"], duration=30)
# print(satus_list)
# if all(satus_list) :
#      print('Aslom')

# ##--------------------------------------------------------
def display_streaming(cameraName):
    
    """
    Generates frames from a video feed and streams them.

    Args:
        cameraName (str): Name of the camera.

    Yields:
        bytes: Encoded frame data.
    """
    query = {'Camera Name': cameraName}

    try:
        src = int(find_existing_document(db['CameraInfo'], query)['Port'])
    except Exception:
        src = str(find_existing_document(db['CameraInfo'], query)['Link'])

    srcType = find_existing_document(db['CameraInfo'], query)['Source Type']

    print(src, srcType)

    cap, fps = readSource(srcType, src)

    if cap is None:
        print("Error: Capture object is None.")
        return 

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            else:
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
    except Exception as e:
        print(f"Error occurred during frame streaming: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
