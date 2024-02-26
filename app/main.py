from flask import Flask , render_template , request , redirect , url_for, jsonify 
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user , current_user
from functions import *           
from threading import Thread
from werkzeug.security import check_password_hash , generate_password_hash
from flask_cors import CORS
from bson import ObjectId
import os 
import warnings
from MongoPackageV2 import *

warnings.filterwarnings("ignore")



# Initialize Flask application
app = Flask(__name__ )

#____________________________________________________________
# Configure secret key for session management
app.config['SECRET_KEY'] = os.urandom(32)

#____________________________________________________________
# Enable Cross-Origin Resource Sharing (CORS) for handling requests from different origins
CORS(app)

#____________________________________________________________
@app.route('/')
def hello_world():
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
        user_obj = User(user_id=user['_id'], username=user['Username'], email=user['Email'], phone=user.get('Phone', ''))  
        login_user(user_obj)
        return jsonify({"username": username, "token": "token", "logined": True})
    else:
        return jsonify({"mess": "Invalid username or password"})

# ##--------------------------------------
@app.route('/signout', methods=['GET'])
@login_required
def signout():
    logout_user()
    return redirect(url_for('signin'))

#____________________________________________________________
##################### STATIC LIST APIs ######################
#____________________________________________________________
@app.route('/get_cameraname', methods=['GET'])
def get_cameraname():
    cameraName = finding_camera_names()
    # cameraName = [1,2,4]
    return jsonify(cameraName)

# ##--------------------------------------
@app.route('/get_modelname', methods=['GET'])
def get_modelname():
    modelname = ['violence' , 'vehicle' , 'crowdedDensity' , 'crossingBorder' , 'crowded' ,'gender']
    return jsonify(modelname)

# ##--------------------------------------
@app.route('/get_sourcetype', methods=['GET'])
def get_sourcetype():
    sourcetype = ['WEBCAM' , 'RTSP' , 'URL']
    return jsonify(sourcetype)

#____________________________________________________________
#################### MAIN FUNCTIONALITY #####################
#____________________________________________________________
@app.route('/addCamera', methods=['POST'])
# @login_required
def add_camera():
    data = request.form
    if not data:
        return jsonify({'mess': "No camera data provided" })

    camera_name = data.get('cameraName')
    source_type = data.get('sourceType')
    source = data.get('source')
    
    if not all([camera_name, source_type, source]):
        return jsonify({'mess': "Incomplete camera data provided" })

    insert_camera_info(camera_name, source_type, source)
    return jsonify({'cameraName': camera_name, 'sourceType': source_type, 'source': source})

#____________________________________________________________
@app.route('/applyModel', methods=['GET', 'POST'])
# @login_required
def apply_Model():
    data = request.form
    if not data:
        return jsonify({'mess': "No model data provided" })

    camera_name = data.get('cameraname')
    model_name = data.get('modelname')
    print(type(model_name))
    print(model_name)
    model_name = [name.strip().strip('"') for name in model_name.strip("[]").replace("'", "").split(",")]
    print(type(model_name))
    print(model_name)

    if not all([camera_name, model_name]):
        return jsonify({'mess': "Incomplete model data provided" })
    
    multiModelRunInsert(camera_name, model_name)

    return jsonify({'cameraName': camera_name, 'modelName/s': model_name})


#____________________________________________________________
###################### QUERY FUNCTIONS ######################
#____________________________________________________________
@app.route('/camcountavg', methods=['GET', 'POST']) 
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
#@login_required
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
#@login_required
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
#@login_required
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
@app.route('/postcam_getallmodelsStat', methods =['POST','GET'])
#@login_required
def postcam_getallmodelsStat_route():
    data = request.form
    if not data:
        return jsonify({'mess': "No data provided" }) 
    
    cameraName = data.get('cameraname')
    dicRes = postcam_getallmodelsStat(cameraName)
    # print(dicRes)
    # dictionary = dicRes.to_dict(orient= 'records')
    # # print(dictionary)

    return jsonify(dicRes) 

#____________________________________________________________
@app.route('/postcam_geteachmodelperH', methods =['POST','GET'])
#@login_required
def postcam_geteachmodelperH_route():
    data = request.form
    if not data:
        return jsonify({'mess': "No data provided" }) 
    
    cameraName = data.get('cameraname')
    day = data.get('day')
    month = data.get('month')
    year = data.get('year')
    dicRes = postcam_geteachmodelperH(cameraName, day, month, year)
    dictionary = dicRes.to_dict(orient= 'records')

    return jsonify(dictionary)  



#____________________________________________________________
@app.route('/postcam_getallmodelsperH', methods =['POST','GET'])
#@login_required
def postcam_getallmodelsperH_route():
    data = request.form
    if not data:
        return jsonify({'mess': "No data provided" }) 
    
    cameraName = data.get('cameraname')
    day = data.get('day')
    month = data.get('month')
    year = data.get('year')
    dicRes = postcam_getallmodelsperH(cameraName, day, month, year)
    # print(dicRes)
    dictionary = dicRes.to_dict(orient= 'records')
    # print(dictionary)

    return jsonify(dictionary)  










#____________________________________________________________
@app.route('/eachmodelstatistics' , methods = ['POST'])
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




#____________________________________________________________
@app.route('/getallmodelsincam', methods=['POST'])
def getallmodelsincam():
    data = request.form
    if not data:
        return jsonify({'mess': "No data provided" }) 
    
    cameraname = data.get('cameraname')
    dictionary = all_camera_info(cameraname)
    return jsonify(dictionary)



#____________________________________________________________
@app.route('/getallcamsinmodel', methods=['POST'])
def getallcamsinmodel():
    data = request.form
    if not data:
        return jsonify({'mess': "No data provided" }) 
    
    modelname = data.get('modelname')
    dictionary = all_cameras_in_model(modelname)
    return jsonify(dictionary)




#____________________________________________________________
# collection = db['CameraInfo']
# def watch_changes():
#     change_stream = collection.watch(full_document='updateLookup') 
#     for change in change_stream:
#         if change['operationType'] == 'insert':
#             print("New Camera Name Inserted:", change['fullDocument']['Camera Name'])
#         elif change['operationType'] == 'delete':
#             print("Document Deleted:", change['documentKey']['_id'])

@app.route('/camera_names'  ,methods=['GET', 'POST'])
def get_camera_names():
    cam_list = finding_camera_names()
    print(cam_list)
    return jsonify(cam_list)

#____________________________________________________________
if __name__ == '__main__':
    # Start change stream watcher in a separate thread
    # change_stream_thread = Thread(target=watch_changes)
    # change_stream_thread.start()
    app.run(host='0.0.0.0',port=8080 , debug=True)