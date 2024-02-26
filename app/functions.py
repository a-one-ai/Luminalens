from pytube import YouTube 
import streamlink   
import cv2  
from projectModel import *  
from MongoPackageV2 import *  
import warnings
warnings.filterwarnings("ignore")

global capture  

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

# ##----------------------------------------
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

# ##----------------------------------------
def readSource(srcType, src):
    """
    Reads the video source and initializes a video capture object.

    Args:
        srcType (str): Type of the video source ('WEBCAM', 'RTSP', or 'URL').
        src (str): Source identifier (e.g., port number for 'WEBCAM' and 'RTSP', URL for 'URL').

    Returns:
        cv2.VideoCapture: Video capture object.
    """
    global capture 
    try: 
        if srcType == 'WEBCAM': 
            capture = cv2.VideoCapture(src) 
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
                capture = cv2.VideoCapture(vsrc) 
                fps = capture.get(cv2.CAP_PROP_FPS)                 
                
            except Exception as e: 
                print(f"Error in capturing YouTube video: {e}") 
                vsrc = stream(src) 
                capture = cv2.VideoCapture(vsrc) 
                fps = 30 

        return capture , fps     
                            
    except Exception as e: 
        print(f"Error in readSource: {e}") 
        capture = None 

        return capture 

# #_________________________________________________________
def process_frame(frame, modelName , cameraName):
    """
    Process a single frame using the specified model.

    Args:
        frame (numpy.ndarray): Input frame.
        modelName (str): Name of the model to process the frame.

    Returns:
        tuple: (path, res) - Path of the processed image and result from the model.
    """
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
        path, res = crossingBorder(frame,cameraName)
        return path, res

    elif modelName == 'crowded':
        path, res = crowded(frame)
        return path, res 

    elif modelName == 'gender':
        path, res = detect_GENDER(frame,cameraName)
        return path, res
        

# ##----------------------------------------
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

# ##----------------------------------------
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
def run_models(modelName, frame, cameraName):
    """
    Runs specified models on a single frame obtained from a video feed.

    Args:
        modelName (str or list of str): Name of the model(s) to process frames.
        frame (numpy.ndarray): Single frame obtained from the video feed.
        cameraName (str): Name of the camera.

    Returns:
        list: List of tuples containing the path of the processed image, result from each model,
              camera name, and model name.
    """
    results = []

    for model in modelName:
        if model == 'violence':
            path, res = violence(frame  , cameraName)
            results.append((path, res, cameraName, model))


        elif model == 'vehicle':
            path, res = vehicleCounting(frame  , cameraName)
            results.append((path, res, cameraName, model))


        elif model == 'crowdedDensity':
            path, res = crowdedDensity(frame  , cameraName)
            results.append((path, res, cameraName, model))


        elif model == 'crossingBorder':
            path, res = crossingBorder(frame , cameraName)
            results.append((path, res, cameraName, model))


        elif model == 'crowded':
            path, res = crowded(frame  , cameraName)
            results.append((path, res, cameraName, model))


        elif model == 'gender':
            path, res = detect_GENDER(frame  , cameraName)
            results.append((path, res, cameraName, model))

    return results

# ##----------------------------------------
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
    for model in selected_models:
        results = run_models([model], frame, cameraName)
        all_results.extend(results)
    return all_results

# ##----------------------------------------
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
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1  
        if  (count % int(fps) == 0) or (count == 1):     
                all_results = run_selected_models(modelNames, frame, cameraName)
                yield all_results

    
        if cv2.waitKey(27) & 0xFF == ord('q'): 
            break 

    cap.release()
    cv2.destroyAllWindows()

# ##----------------------------------------
def multiModelRunInsert(cameraName, modelNames):
    """
    Processes video frames from a camera for multiple specified models simultaneously and inserts the results into a database.

    Args:
        cameraName (str): Name of the camera.
        modelNames (list of str): List of model names to process frames.
    """
    try:
        results_generator = videoFeedMulti(cameraName, modelNames)
        for results in results_generator:
            for result in results:
                path, res, cameraName, modelName = result
                insert_model_info(cameraName, modelName, res, path)
    except Exception as e:
        print(f"Error in multiModelRunInsert: {e}")
    finally:
        # Release resources
        cv2.destroyAllWindows()
