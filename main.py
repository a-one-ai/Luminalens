from flask import Flask , render_template , request , redirect , url_for, jsonify , Response , make_response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user , current_user
from functions import *           
from threading import Thread
from werkzeug.security import check_password_hash , generate_password_hash
from flask_cors import CORS
from bson import ObjectId
import os 
import warnings
from MongoPackageV2 import *
from functools import wraps
import jwt
import secrets
import psutil
import time
warnings.filterwarnings("ignore")



# Initialize Flask application
app = Flask(__name__ )

#____________________________________________________________
# Configure secret key for session management
SECRET_KEY = secrets.token_hex(16)
app.config['SECRET_KEY'] = SECRET_KEY

#____________________________________________________________
# Enable Cross-Origin Resource Sharing (CORS) for handling requests from different origins
CORS(app)

#____________________________________________________________
# def generate_token(user_id):
#     user_id_str = str(user_id)
#     payload = {'user_id': user_id_str}
#     token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
#     return token



def generate_token(user_id, expiration_time=3600 *10):  
    user_id_str = str(user_id)
    expiration_timestamp = int(time.time()) + expiration_time  
    payload = {
        'user_id': user_id_str,
        'exp': expiration_timestamp  
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
    return token
# ##--------------------------------------
# Add debug print statements to verify token decoding
def verify_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])

        # Debug print statement to verify decoded payload
        print("Decoded payload:", payload)

        return payload['user_id']
    except jwt.ExpiredSignatureError:
        # Token has expired
        print("Token has expired")
        return None
    except jwt.InvalidTokenError:
        # Token is invalid
        print("Invalid token")
        return None
# ##--------------------------------------
def token_required(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        token = request.headers.get('token')

        print("Received token:", token)

        if not token:
            print("Token is missing")
            return jsonify({'message': 'Token is missing'}), 401

        user_id = verify_token(token)
        if not user_id:
            print("Invalid token")
            return jsonify({'message': 'Invalid token'}), 401

        return func(*args, **kwargs)

    return decorated


#____________________________________________________________
logs_collection = db["systemLogs"]


def log_route_info_to_db(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        log_entry = {
            'timestamp': datetime.now(),
            'route': request.url,
            'method': request.method,
            'parameters': dict(request.form),
            'message': f"Route: {request.url}, Method: {request.method}, Parameters: {request.args.to_dict()}"
        }

        response = func(*args, **kwargs)

        if isinstance(response, tuple):  
            status_code = response[1]
            data = 'No Data, Login required' if status_code == 302 else ''
            response = Response(data, status=status_code)
        else:
            status_code = response.status_code
            data = '' if status_code == 302 else response.data.decode('utf-8')

        response_info = {
            'status_code': status_code,
            'data': data
        }
        log_entry['response'] = response_info

        logs_collection.insert_one(log_entry)

        return response
    return wrapper

#____________________________________________________________

    return render_template('base.html')
#____________________________________________________________
################## AUTHENTICATION FUNCTIONS #################
#____________________________________________________________
login_manager = LoginManager(app)
login_manager.login_view = 'signin'

# ##--------------------------------------
from flask_login import UserMixin

class User(UserMixin):
    def __init__(self, user_id, username, email, phone):
        self.id = user_id 
        self.username = username
        self.email = email
        self.phone = phone

    def get_id(self):
        return str(self.id)  

# ##--------------------------------------
@login_manager.user_loader
def load_user(user_id):
    user_data = db['users'].find_one({'_id': ObjectId(user_id)})
    if user_data:
        username = user_data.get('Username')
        email = user_data.get('Email')
        phone = user_data.get('Phone', '') 
        user = User(user_id, username, email, phone)
        return user
    return None
# ##--------------------------------------
@login_manager.unauthorized_handler
def unauthorized_callback():
    return redirect(url_for('signin'))
#____________________________________________________________

@app.route('/signup', methods=['POST'])
def signup():

    data = request.form
    if not data:
        return jsonify({'mess': "No data provided" })

    username = data.get('username')
    email = data.get('email')
    phone = data.get('phone')
    password = data.get('password')

    if not username or not email or not phone or not password:
        return jsonify({'mess': "All fields are required" })

    users_collection = db['users']
    hashed_password = generate_password_hash(password)
    new_user = {
        'Username': username,
        'Email': email,
        'Phone': phone,
        'Password': hashed_password
    }
    result = users_collection.insert_one(new_user)
    if result.inserted_id:
        response = jsonify({'message': 'Account created successfully!'})
        response.status_code = 200
        return redirect(url_for('signin'))
    else:
        response = jsonify({'message': 'Account fails to be created!'})
        response.status_code = 500
        return redirect(url_for('signin'))

# ##--------------------------------------
@app.route('/signin', methods=['POST'])
@log_route_info_to_db
def signin():

    data = request.form
    if not data:
        return jsonify({'mess': "No data provided" })
    
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'mess': "Username and password must be provided" })
    
    user = db['users'].find_one({'Username': username})
    
    if user and check_password_hash(user['Password'], password):
        user_id=user['_id']
        user_obj = User(user_id=user['_id'], username=user['Username'], email=user['Email'], phone=user.get('Phone', ''))  
        login_user(user_obj)
        token = generate_token(user_id)
        return jsonify({"username": username, "token": token, "logined": True})
    else:
        return jsonify({"mess": "Invalid username or password"})
    
# ##--------------------------------------
@app.route('/signout', methods=['GET'])
@log_route_info_to_db
# @login_required
def signout():
    logout_user()
    return redirect(url_for('signin'))

#____________________________________________________________
################/##### STATIC LIST APIs ######################
#____________________________________________________________
@app.route('/get_cameraname', methods=['GET'])
# @token_required
@log_route_info_to_db
# @login_required
def get_cameraname():
    cameraName = finding_camera_names()
    # cameraName = [1,2,4]
    return jsonify(cameraName)

# ##--------------------------------------
@app.route('/get_modelname', methods=['GET'])
# @token_required
@log_route_info_to_db
# @login_required
def get_modelname():
    modelname = ['violence' , 'vehicle' , 'crowdedDensity' , 'crossingBorder' , 'crowded' ,'gender' , 'clothes color' , 'enter exit counting']
    return jsonify(modelname)

# ##--------------------------------------
@app.route('/get_sourcetype', methods=['GET'])
# @token_required
def get_sourcetype():
    sourcetype = ['WEBCAM' , 'RTSP' , 'URL']
    return jsonify(sourcetype)

# ##--------------------------------------

@app.route("/get_colosList" , methods = ['GET'])
def get_colorsList():
    colorsList =[
        'Red'     ,
        'Blue'    ,
        'Green' ,       
        'Yellow' ,   
        'Purple' ,
        'Cyan' ,     
        'Orange' ,
        'Brown',
        'Black' ,
        'White' ]
    return jsonify(colorsList)

# ##--------------------------------------
# @token_required
@app.route('/get_all_cameras_info', methods=['GET'])
def get_all_cameras_info():
    cameras_data = all_cameras_info()
    return jsonify(cameras_data)

# ##--------------------------------------
# @token_required
@app.route('/get_all_cameras_people_count', methods=['GET'])
def get_all_cameras_people_count():
    people_count_data = counting_cameras_average_frist_model()
    return jsonify(people_count_data)


# ##--------------------------------------
# @token_required
@app.route('/get_all_cameras_vechile_count', methods=['GET'])
def get_all_cameras_vechile_count():
    vehcile_count_data = vehcile_counting_cameras_average()
    return jsonify(vehcile_count_data)


# ##--------------------------------------
@app.route('/get_running_now_info', methods=['GET'])
def get_running_now_info():
    running_now_detials = all_running_now_data()
    return jsonify(running_now_detials)


#____________________________________________________________
#################### MAIN FUNCTIONALITY #####################
#____________________________________________________________


@app.route('/addCamera', methods=['POST'])
@log_route_info_to_db
# @token_required
def add_camera():
    data = request.form
    if not data:
        return jsonify({'message': "No camera data provided"}), 400

    camera_name = data.get('cameraName')
    source_type = data.get('sourceType')
    source = data.get('source')

    if not all([camera_name, source_type, source]):
        return jsonify({'message': "Incomplete camera data provided"}), 400

    satus = insert_camera_info(camera_name, source_type, source)
    if satus['Inserted']  :
         return jsonify({'cameraName': camera_name, 'sourceType': source_type, 'source': source ,
                          'inserted' : True , 'Message' :'Camera Inserted Successfully'} ) 
    
    else : 
         return jsonify({'cameraName': camera_name, 'sourceType': source_type, 'source': source ,
                          'inserted' : False, 'Message' :'Camera Already Exists'} ) 
                        

camera_stop_flags = {}
# Function to stop processing for a specific camera
def reset_processing_flag():
    global stop_processing
    stop_processing = False

# Function to reset processing for a specific camera
def reset_processing_for_camera(camera_name):
    global camera_stop_flags
    camera_stop_flags[camera_name] = False  


# Define model_stop_flags as a global variable
model_stop_flags = {}
# ##--------------------------------------------------------
def enable_model_for_camera(camera_name, model_name):
    global model_stop_flags
    model_stop_flags[(camera_name, model_name)] = False


# Function to disable processing for a specific model on a specific camera
# ##--------------------------------------------------------    
def disable_model_for_camera(camera_name, model_name):
    global model_stop_flags
    model_stop_flags[(camera_name, model_name)] = True


#____________________________________________________________
@app.route('/applyModelOld', methods=['GET', 'POST'])
@log_route_info_to_db
# @login_required
def applyModelOld():
    data = request.form
    if not data:
        return jsonify({'mess': "No model data provided" })

    camera_name = data.get('cameraname')
    model_name = data.get('modelname')
    # print(type(model_name))
    # print(model_name)
    model_name = [name.strip().strip('"') for name in model_name.strip("[]").replace("'", "").split(",")]
    # print(type(model_name))
    # print(model_name)

    if not all([camera_name, model_name]):
        return jsonify({'mess': "Incomplete model data provided" })

    reset_processing_flag()
    reset_processing_for_camera(camera_name=camera_name)
    for model in model_name : 
        enable_model_for_camera(camera_name,model)
    multiModelRunInsert(camera_name, model_name)

    return jsonify({'cameraName': camera_name, 'modelName/s': model_name})



def apply_Model(camera_name, model_names):
    reset_processing_flag()
    reset_processing_for_camera(camera_name=camera_name)
    for model in model_names : 
        enable_model_for_camera(camera_name,model)
    multiModelRunInsert(camera_name, model_names)


@app.route('/applyModel', methods=['POST'])
@log_route_info_to_db
def applyModel():
    data = request.form
    if not data:
        return jsonify({'mess': "No model data provided" })

    camera_name = data.get('cameraname')
    model_name = data.get('modelname')
    # print(type(model_name))
    # print(model_name)
    model_name = [name.strip().strip('"') for name in model_name.strip("[]").replace("'", "").split(",")]
    if not all([camera_name, model_name]):
        return jsonify({'mess': "Incomplete model data provided" })

    apply_Model_Testing(cameraName=camera_name, modelNames=model_name, duration=30)
    # print(satus_list)
    if all(satus_list) :
        print('Tested For 30 Second and All is Good')
        # Start the long-running task in a separate thread
        task_thread = Thread(target=apply_Model, args=(camera_name, model_name))
        task_thread.start()
        # Return a response immediately
        return jsonify({'mess': f'Apply Models {model_name} started in the background for {camera_name} camera',
                       'Testing' : 'Testing for 30 seconda and All is Good',
                       'Camera Name' : camera_name,
                       'Models' : model_name })
    else :
        return jsonify({'mess': f'Apply Models {model_name} can\'t started in the background  for {camera_name} camera',
                        'test' : 'Testing for 30 seconda and Error during Tested',
                       'Camera Name' : camera_name,
                       'Models' : model_name })                      

 
#____________________________________________________________
@app.route('/stopallModels', methods=['POST'])
@log_route_info_to_db
def stop_processing_api():
    stop_processing_function()
    if (delete_collection(db['RunningNow'])) :
      print('Deleted RunningNow Collection')
      return jsonify({'message': 'All Models & Cameras stopped.'})
    else : 
      return jsonify({'message': 'Can\'t Stop '})


#____________________________________________________________
# @app.route('/stopProcessingCamera', methods=['POST'])
# @log_route_info_to_db
# def stopProcessingCamera():
#     data = request.form
#     if not data:
#         return jsonify({'mess': "No camera data provided" })

#     camera_name = data.get('cameraname')
#     if not camera_name:
#         return jsonify({'mess': "Incomplete camera data provided" })
    

#     stop_processing_for_camera(camera_name)    
#     # Stop processing for the specific camera
#     if (delete_documents_for_camera(db['RunningNow'], camera_name) ):
#         return jsonify({'message': f'Processing stopped for camera {camera_name}.'})

#     else : 
#       return jsonify({'message': 'Can\'t Stop  Camera'})




@app.route('/stopProcessingCamera', methods=['POST','GET'])
@log_route_info_to_db
def stopProcessingCamera():
    data = request.form
    if not data:
        return jsonify({'mess': "No camera data provided" })

    camera_name = data.get('cameraname')
    if not camera_name:
        return jsonify({'mess': "Incomplete camera data provided" })
    
    stop_processing_for_camera(camera_name)

    return redirect(url_for('DeleteCameraDocs',camera_name = camera_name))

@app.route('/DeleteCameraDocs', methods=['GET'])
def DeleteCameraDocs():
    camera_name = request.args.get('camera_name')
    print(camera_name)
    status_delete = delete_documents_for_camera(db['RunningNow'], camera_name)
    if status_delete == True:
        return jsonify({'message': f'Processing stopped for camera {camera_name}. Deleting Documents: {status_delete}'})
    else:
        return jsonify({'message': f'Processing stopped for camera {camera_name}.  Deleting Documents: {status_delete}'})


#____________________________________________________________
# @app.route('/stopCameraModel', methods=['POST'])
# @log_route_info_to_db
# def disable_camera_model():
#     data = request.form
#     if not data:
#         return jsonify({'mess': "No data provided" })

#     camera_name = data.get('cameraname')
#     model_name = data.get('modelname')
#     model_name = [name.strip().strip('"') for name in model_name.strip("[]").replace("'", "").split(",")]
#     print(type(model_name))
#     print(model_name)    
#     for model in model_name : 
#         disable_model_for_camera(camera_name,model)    
#         print('Model Stopped :', model_name ,' For ', camera_name)
#         # disable_model_for_camera(camera_name, model_name)
#     return jsonify({'message': f"Model {model_name} disabled for camera '{camera_name}'."})

@app.route('/stopCameraModel', methods=['POST'])
@log_route_info_to_db
def disable_camera_model():
    data = request.form
    if not data:
        return jsonify({'mess': "No data provided" })

    camera_name = data.get('cameraname')
    print(camera_name)
    model_name = data.get('modelname')
    model_name = [name.strip().strip('"') for name in model_name.strip("[]").replace("'", "").split(",")]
    print(type(model_name))
    print(model_name)    
    running_now_data = all_running_now_data()
    print(running_now_data)
    for data in running_now_data :
        # print(data)
        if (data['Camera Name'] == camera_name) :
            models_applied = data['Models Applied']
            models_to_remove = model_name
            models_updated_list = [item for item in models_applied if item not in models_to_remove]
            print(models_updated_list)    
            stop_processing_for_camera(camera_name)
            status_delete = delete_documents_for_camera(db['RunningNow'], camera_name)
            print(status_delete)
            if status_delete == True:            
                apply_Model_Testing(cameraName=camera_name, modelNames=models_updated_list, duration=15)
                # print(satus_list)
                if all(satus_list) :
                    print('Tested For 15 Second and All is Good')
                    # Start the long-running task in a separate thread
                    task_thread = Thread(target=apply_Model, args=(camera_name, models_updated_list))
                    task_thread.start()                
                    return jsonify({
                        'Models Run Before' : models_applied ,                              
                       'Removed Models' : models_to_remove,                                    
                       'Running Now Models' : f'{models_updated_list}',
                       'Camera Name' : camera_name ,
                       'mess': f'Apply Models {models_updated_list} started in the background for {camera_name} camera',
                       'test' : 'Testing for 15 seconda and All is Good',

                            })                    
                    
                else :
                    return jsonify({
                        'Models Run Before' : models_applied ,                              
                       'Removed Models' : models_to_remove,                                    
                       'Running Now Models' : f'{models_updated_list}',
                       'Camera Name' : camera_name ,
                        'mess': f'Apply Models {models_updated_list} cannot in the background for {camera_name} camera',
                        'test' : 'Testing for 15 seconds and Error during Tested',
                            })      
            
            else :
                return jsonify({'mess' : 'Unexpected Error Canot Delete From RunningNow Collection'})      
        
                     
@app.route('/app_resources', methods=['GET'])
def app_resources():
    try:
        app_cpu_percent = psutil.Process().cpu_percent(interval=None)
        app_memory_percent = psutil.Process().memory_percent()

        return jsonify(app_cpu_percent=app_cpu_percent, app_memory_percent=app_memory_percent), 200
    except Exception as e:
        return jsonify(error=str(e)), 500



#____________________________________________________________
###################### QUERY FUNCTIONS ######################
#____________________________________________________________
@app.route('/camcountavg', methods=['GET', 'POST']) 
@log_route_info_to_db
# @login_required
def camcountavg(): 
    data = request.form 
    if not data: 
        return jsonify({'mess': "No data provided" }) 

    camera_name = data.get('cameraname') 
    model_name = data.get('modelname') 
    if not all([camera_name, model_name]): 
        return jsonify({'mess': "Incomplete data" }) 

    avg = average_camera_count(camera_name ,model_name ) 
    return jsonify(avg)


#____________________________________________________________
@app.route('/camcountperH', methods =['POST'])
@log_route_info_to_db
# @login_required
def camcountperH():
    data = request.form
    if not data:
        return jsonify({'mess': "No data provided" }) 
    
    cameraName = data.get('cameraname')
    modelName = data.get('modelname')
    day = data.get('day')
    month = data.get('month')
    year = data.get('year')
    dicRes = date_filter_aggerigates_df(cameraName, modelName, day, month, year)
    dictionary = dicRes.to_dict(orient= 'records')
    return jsonify(dictionary)

#____________________________________________________________
@app.route('/camcountperM', methods =['POST','GET'])
@log_route_info_to_db
# @login_required
def camcountperM():
    data = request.form
    if not data:
        return jsonify({'mess': "No data provided" }) 
    
    cameraName = data.get('cameraname')
    modelName = data.get('modelname')
    month = data.get('month')
    year = data.get('year')
    dicRes = months_filter_aggerigates_df(cameraName, modelName, month, year)
    dictionary = dicRes.to_dict(orient= 'records')
    return jsonify(dictionary)    

#____________________________________________________________
@app.route('/camcountperY', methods =['POST','GET'])
@log_route_info_to_db
# @login_required
def camcountperY():
    data = request.form
    if not data:
        return jsonify({'mess': "No data provided" }) 
    
    cameraName = data.get('cameraname')
    modelName = data.get('modelname')
    year = data.get('year')
    dicRes = year_filter_aggerigates_df(cameraName, modelName, year)
    dictionary = dicRes.to_dict(orient= 'records')

    return jsonify(dictionary)    

#____________________________________________________________
# @app.route('/camcountall', methods =['POST','GET'])
# #@login_required
# def camcountall():
#     data = request.form
#     if not data:
#         return jsonify({'mess': "No data provided" }) 
    
#     cameraName = data.get('cameraname')
#     day = data.get('day')
#     month = data.get('month')
#     year = data.get('year')
#     dicRes = date_filter_aggerigates_df_allmodels(cameraName, day, month, year)
#     print(dicRes)
#     dictionary = dicRes.to_dict(orient= 'records')
#     print(dictionary)

#     return jsonify(dictionary)

#____________________________________________________________
@app.route('/postcam_geteachmodelstat', methods=['POST'])
@log_route_info_to_db
#@login_required
def postcam_geteachmodelstat_route():
    data = request.form
    if not data:
        return jsonify({'mess': "No data provided"})
    
    cameraName = data.get('cameraname')
    dicRes = postcam_geteachmodelstat(cameraName)
    dictionary = dicRes.to_dict(orient='records')

    return jsonify(dictionary)

#____________________________________________________________
# @app.route('/postcam_getallmodelsStat', methods =['POST','GET'])
# @log_route_info_to_db
# #@login_required
# def postcam_getallmodelsStat_route():
#     data = request.form
#     if not data:
#         return jsonify({'mess': "No data provided" }) 
    
#     cameraName = data.get('cameraname')
#     dicRes = postcam_getallmodelsStat(cameraName)
#     # print(dicRes)
#     # dictionary = dicRes.to_dict(orient= 'records')
#     # # print(dictionary)

#     return jsonify(dicRes) 


#____________________________________________________________
# @app.route('/postcam_geteachmodelperH', methods =['POST','GET'])
# @log_route_info_to_db
# #@login_required
# def postcam_geteachmodelperH_route():
#     data = request.form
#     if not data:
#         return jsonify({'mess': "No data provided" }) 
    
#     cameraName = data.get('cameraname')
#     day = data.get('day')
#     month = data.get('month')
#     year = data.get('year')
#     dicRes = postcam_geteachmodelperH(cameraName, day, month, year)
#     dictionary = dicRes.to_dict(orient= 'records')

#     return jsonify(dictionary)  



#____________________________________________________________
# @app.route('/postcam_getallmodelsperH', methods =['POST','GET'])
# @log_route_info_to_db
# #@login_required
# def postcam_getallmodelsperH_route():
#     data = request.form
#     if not data:
#         return jsonify({'mess': "No data provided" }) 
    
#     cameraName = data.get('cameraname')
#     day = data.get('day')
#     month = data.get('month')
#     year = data.get('year')
#     dicRes = postcam_getallmodelsperH(cameraName, day, month, year)
#     # print(dicRes)
#     dictionary = dicRes.to_dict(orient= 'records')
#     # print(dictionary)

    # return jsonify(dictionary)  

@app.route('/postmanvehicle', methods =['POST','GET'])
@log_route_info_to_db
#@login_required
def postcam_vehicle_route():
    data = request.form
    if not data:
        return jsonify({'mess': "No data provided" }) 
    
    cameraName = data.get('cameraname')
    dicRes = postcam_getvechile(cameraName)
    # print(dicRes)
    # dictionary = dicRes.to_dict(orient= 'records')
    # # print(dictionary)

    return jsonify(dicRes) 

#____________________________________________________________
@app.route('/eachmodelstatistics' , methods = ['POST'])
@log_route_info_to_db
#@login_required
def eachmodelstatistics():
        data = request.form
        if not data:
            return jsonify({'mess': "No data provided" }) 
        
        cameraName = data.get('cameraname')
        day = data.get('day')
        month = data.get('month')
        year = data.get('year')


        dictionary = None 

        if  cameraName and day and month and year :
            dictionary = postcam_geteachmodelperH(cameraName, day, month, year)
            # print(dictionary)
            dictionary = dictionary.to_dict(orient='records')
            # print(dictionary)
            return jsonify(dictionary) 



        elif cameraName and not day and month and year:
            dictionary = postcam_geteachmodelperM(cameraName, month, year)
            # print(dictionary)
            dictionary = dictionary.to_dict(orient='records')
            # print(dictionary)
            return jsonify(dictionary) 



        elif cameraName and not day and not month and year:
            dictionary = postcam_geteachmodelperY(cameraName, year)
            # print(dictionary)
            dictionary = dictionary.to_dict(orient='records')
            # print(dictionary)
            return jsonify(dictionary) 


#____________________________________________________________
@app.route('/allmodelsstatistics' , methods = ['POST'])
@log_route_info_to_db
#@login_required
def allmodelsstatistics():
        data = request.form
        if not data:
            return jsonify({'mess': "No data provided" }) 
        
        cameraName = data.get('cameraname')
        day = data.get('day')
        month = data.get('month')
        year = data.get('year')


        dictionary = None 

        if  cameraName and day and month and year :
            dictionary = postcam_getallmodelsperH(cameraName, day, month, year)
            # print(dictionary)
            try :
                dictionary = dictionary.to_dict(orient='records')
                return jsonify(dictionary) 
            except :
                return jsonify(dictionary)
            # print(dictionary)




        elif cameraName and not day and month and year:
            dictionary = postcam_getallmodelsperM(cameraName, month, year)
            # print(dictionary)
            try :
                dictionary = dictionary.to_dict(orient='records')
                return jsonify(dictionary) 
            except :
                return jsonify(dictionary)



        elif cameraName and not day and not month and year:
            dictionary = postcam_getallmodelsperY(cameraName, year)
            # print(dictionary)
            try :
                dictionary = dictionary.to_dict(orient='records')
                return jsonify(dictionary) 
            except :
                return jsonify(dictionary)
            
#______________________________________________________________
@app.route('/people_count_time_range' , methods = ['POST'])
@log_route_info_to_db
#@login_required
def people_count_time_range():
        data = request.form
        if not data:
            return jsonify({'mess': "No data provided" }) 
        
        day = data.get('day')
        month = data.get('month')
        year = data.get('year')


        data = None 

        if  day and month and year :
            data = get_all_cameras_count_perH(day, month, year)
            # print(dictionary)
            try :
                return jsonify(data) 
            except :
                return jsonify(data)
            # print(dictionary)




        elif  not day and month and year:
            data = get_all_cameras_count_perM(month, year)
            # print(dictionary)
            try :
                return jsonify(data) 
            except :
                return jsonify(data)



        elif   not day and not month and year:
            data = get_all_cameras_count_perY(year)
            # print(dictionary)
            try :
                return jsonify(data) 
            except :
                return jsonify(data)        

#______________________________________________________________
@app.route('/gender_count_time_range' , methods = ['POST'])
@log_route_info_to_db
#@login_required
def gender_count_time_range():
        data = request.form
        if not data:
            return jsonify({'mess': "No data provided" }) 
        
        day = data.get('day')
        month = data.get('month')
        year = data.get('year')


        data = None 

        if  day and month and year :
            data = get_all_cameras_genderPerH(day, month, year)
            # print(dictionary)
            try :
                return jsonify(data) 
            except :
                return jsonify(data)
            # print(dictionary)




        elif  not day and month and year:
            data = get_all_cameras_genderPerM(month, year)
            # print(dictionary)
            try :
                return jsonify(data) 
            except :
                return jsonify(data)



        elif   not day and not month and year:
            data = get_all_cameras_genderPerY(year)
            # print(dictionary)
            try :
                return jsonify(data) 
            except :
                return jsonify(data)        

#______________________________________________________________
@app.route('/vechile_count_time_range' , methods = ['POST'])
@log_route_info_to_db
#@login_required
def vechile_count_time_range():
        data = request.form
        if not data:
            return jsonify({'mess': "No data provided" }) 
        
        day = data.get('day')
        month = data.get('month')
        year = data.get('year')


        data = None 

        if  day and month and year :
            data = get_all_vechile_count_perH(day, month, year)
            # print(dictionary)
            try :
                return jsonify(data) 
            except :
                return jsonify(data)
            # print(dictionary)




        elif  not day and month and year:
            data = get_all_vechile_count_perM(month, year)
            # print(dictionary)
            try :
                return jsonify(data) 
            except :
                return jsonify(data)



        elif   not day and not month and year:
            data = get_all_vechile_count_perY(year)
            # print(dictionary)
            try :
                return jsonify(data) 
            except :
                return jsonify(data)            


@app.route('/allmodelsstatisticsvehicle' , methods = ['POST'])
@log_route_info_to_db
#@login_required
def allmodelsstatisticsvehicle():
        data = request.form
        if not data:
            return jsonify({'mess': "No data provided" }) 
        
        cameraName = data.get('cameraname')
        day = data.get('day')
        month = data.get('month')
        year = data.get('year')


        dictionary = None 

        if  cameraName and day and month and year :
            dictionary = postcam_getvechileH(cameraName, day, month, year)
            # print(dictionary)
            try :
                dictionary = dictionary.to_dict(orient='records')
                return jsonify(dictionary) 
            except :
                return jsonify(dictionary)
            # print(dictionary)




        elif cameraName and not day and month and year:
            dictionary = postcam_getvechileM(cameraName, month, year)
            # print(dictionary)
            try :
                dictionary = dictionary.to_dict(orient='records')
                return jsonify(dictionary) 
            except :
                return jsonify(dictionary)



        elif cameraName and not day and not month and year:
            dictionary = postcam_getvechileY(cameraName, year)
            # print(dictionary)
            try :
                dictionary = dictionary.to_dict(orient='records')
                return jsonify(dictionary) 
            except :
                return jsonify(dictionary)



#____________________________________________________________
@app.route('/getallmodelsincam', methods=['POST'])
@log_route_info_to_db
# @login_required
def getallmodelsincam():
    data = request.form
    if not data:
        return jsonify({'mess': "No data provided" }) 
    
    cameraname = data.get('cameraname')
    dictionary = all_camera_info(cameraname)
    return jsonify(dictionary)


#____________________________________________________________
@app.route('/getallcamsinmodel', methods=['POST'])
@log_route_info_to_db
# @login_required
def getallcamsinmodel():
    data = request.form
    if not data:
        return jsonify({'mess': "No data provided" }) 
    
    modelname = data.get('modelname')
    dictionary = all_cameras_in_model(modelname)
    return jsonify(dictionary)

#____________________________________________________________
@app.route("/colorDocs" , methods = ['POST'])
@log_route_info_to_db
# @login_required
def coloeDocs():
    data = request.form 
    if not data :
        return jsonify({"mess" : "No data provided"})
    
    color = data.get('color')
    cameraname = data.get('cameraname')
    docs = clothes_filtering(cameraname , color)

    return jsonify(docs)

#____________________________________________________________
@app.route("/genderCount_Docs" , methods = ['POST'])
@log_route_info_to_db
# @login_required
def genderCount_Docs():
        data = request.form
        if not data:
            return jsonify({'mess': "No data provided" }) 
        
        cameraName = data.get('cameraname')
        day = data.get('day')
        month = data.get('month')
        year = data.get('year')


        Male = None 
        Female = None 
        docs = None 

        if  cameraName and day and month and year :
            docs = gender_filtering_date_aggrigates(cameraName, day, month, year)
            try :
                docs = docs.to_dict(orient='records')
                return jsonify(docs) 
            except :
                return jsonify(docs)



        elif cameraName and not day and month and year:
            docs  = gender_filtering_month_aggregates(cameraName, month, year)
            try :
                docs = docs.to_dict(orient='records')
                return jsonify(docs) 
            except :
                return jsonify(docs)


        elif cameraName and not day and not month and year:
            docs = gender_filtering_year_aggregates(cameraName, year)
            try :
                docs = docs.to_dict(orient='records')
                return jsonify(docs) 
            except :
                return jsonify(docs)

#____________________________________________________________


@app.route("/violence_Docs" , methods = ['POST'])
@log_route_info_to_db
def violence_Docs():
        data = request.form
        if not data:
            return jsonify({'mess': "No data provided" }) 
        print(data)
        cameraName = data.get('cameraname')
        day = data.get('day')
        month = data.get('month')
        year = data.get('year')

 
        docs = None 
        print(cameraName)
        print(day)
        
        if  cameraName and day and month and year :
            docs = VoilenceFilteringH(cameraName, day, month, year)
            # print(type(docs))
            # pprint.pprint(docs)
            return jsonify(docs) 



        elif cameraName and not day and month and year:
            docs  = VoilenceFilteringM(cameraName, month, year)
            return jsonify(docs)



        elif cameraName and not day and not month and year:
            docs = VoilenceFilteringY(cameraName, year)
            return jsonify(docs)


@app.route('/filter_all_cameras_in_violence_bydate',methods=['POST']) 
@log_route_info_to_db
def violence_filter():
    data = request.form
    if not data:
        return jsonify({'mess': "No data provided" }) 
    print(data)
    day = data.get('day')
    month = data.get('month')
    year = data.get('year')
    newdoc =[]
    camera_names = all_cameras_in_violence()
    print(camera_names)
    for cameraName in camera_names:
        if  cameraName and day and month and year :
            docs = VoilenceFilteringH(cameraName, day, month, year)
            newdoc.append(docs)
        elif cameraName and not day and month and year:
            docs  = VoilenceFilteringM(cameraName, month, year)
            newdoc.append(docs)        

        elif cameraName and not day and not month and year:
            docs = VoilenceFilteringY(cameraName, year)
            newdoc.append(docs)

    return jsonify(newdoc)


@app.route('/display_stream'  ,methods=['POST','GET'])
def display_stream() :
    
    data = request.args
    #data = request.form 
    print(data)
    if not data:
             return jsonify({'mess': "No data provided" }) 
    cameraName = data.get('cameraname') 
    reset_processing_flag()        
    #cameraName = 'aslom'
    try :
        return Response(display_streaming(cameraName),mimetype='multipart/x-mixed-replace; boundary=frame')
    except :
        return jsonify({'mess' : 'Can\'t Display'})

    
#____________________________________________________________
#____________________________________________________________


#____________________________________________________________
def streams_inseration(model_camera_info):
        
        collection = db['RunningNow']    
        collection.insert_one(model_camera_info)

def watch_changes():
    try:
        with db.watch(full_document='updateLookup') as change_stream:
            for change in change_stream:
                try:
                    if change['operationType'] == 'insert':
                        RunningNowDic = {
                            "Camera Name": change['fullDocument'].get('Camera Info', {}).get('Camera Name'),
                            "Model Name": change['fullDocument'].get('Model Name')
                        }
                        if RunningNowDic["Camera Name"] and RunningNowDic["Model Name"]:
                            # Define the query to check for existing document
                            query = {
                                "Camera Name": RunningNowDic["Camera Name"],
                                "Model Name": RunningNowDic["Model Name"]
                            }
                            # Check if the document already exists
                            if not check_existing_document(db['RunningNow'], query):
                                # Save the camera info
                                streams_inseration(RunningNowDic)

                except Exception as e:
                    print(f"Error processing change: {e}")
                    continue
    finally:
        # Drop the collection after processing is complete
        collection = db['RunningNow']
        collection.drop()
        

@app.route('/camera_names'  ,methods=['GET', 'POST'])
def get_camera_names():
    cam_list = finding_camera_names()
    print(cam_list)
    return jsonify(cam_list)

#____________________________________________________________
if __name__ == '__main__':
    # Start change stream watcher in a separate thread
    change_stream_thread = Thread(target=watch_changes)
    change_stream_thread.start()
    app.run(host='0.0.0.0',port=8080 , debug=False)    

    # change_stream_thread.join()        
