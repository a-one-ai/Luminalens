from pymongo import MongoClient
from datetime import datetime
import pandas as pd
import pytz
import math
from pymongo.errors import OperationFailure
from time import sleep
import calendar
import pprint


# Global variables for client and database connection
global client, db

# Connect to the default local MongoDB instance and assign the client and database as global
client = MongoClient('mongodb://localhost:27017/')
db = client.CameraProject2


#____________________________________________________________
def delete_collection(existing_collection):
    """
    Delete an entire collection.

    Args:
        existing_collection: Collection to be deleted.

    Returns:
        bool: True if the collection is successfully deleted, False otherwise.
    """
    try:
        existing_collection.drop()
        return True
    except Exception as e:
        print(f"Error deleting collection: {e}")
        return False

#____________________________________________________________
def delete_documents_for_camera(existing_collection, camera_name):
    """
    Delete documents for a specific camera from a given collection.

    Args:
        existing_collection: Collection from which to delete documents.
        camera_name (str): Name of the camera for which documents should be deleted.

    Returns:
        int: The number of deleted documents.
    """
    try:
        
        query = {"Camera Name": camera_name}
        result = existing_collection.delete_many(query)
        deleted_count = result.deleted_count
        if deleted_count > 0 :
            return True
        else :
            return False
    except Exception as e:
        print(f"Error Deleting Documents : {e}")     
        return False

# print(delete_documents_for_camera(db['RunningNow'],'aslom'))


def delete_documents_for_camera_model(existing_collection, camera_name, model_name):
    """
    Delete documents for a specific camera and model from a given collection.

    Args:
        existing_collection: Collection from which to delete documents.
        camera_name (str): Name of the camera for which documents should be deleted.
        model_name (str): Name of the model for which documents should be deleted.

    Returns:
        int: The number of deleted documents.
    """
    try:
        query = {"Camera Name": camera_name, "Model Name": model_name}
        result = existing_collection.delete_many(query)
        deleted_count = result.deleted_count
        if deleted_count > 0:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error Deleting Documents: {e}")     
        return False

#____________________________________________________________
def check_existing_document(existing_collection, query):
    """
    Check if a document exists in a given collection based on the provided query.

    Args:
        existing_collection: Collection to check for document existence.
        query: Query to find the document.

    Returns:
        bool: True if the document exists, False otherwise.
    """
    return existing_collection.find_one(query) is not None


#____________________________________________________________
def find_existing_document(existing_collection, query):
    """
    Find an existing document in a given collection based on the provided query.

    Args:
        existing_collection: Collection to search for the document.
        query: Query to find the document.

    Returns:
        dict: The found document if it exists, None otherwise.
    """
    return existing_collection.find_one(query)

#____________________________________________________________
def update_existing_document(existing_collection, query, update_data):
    """
    Update an existing document in a given collection based on the provided query.

    Args:
        existing_collection: Collection to update the document.
        query: Query to find the document.
        update_data: Data to update in the document.

    Returns:
        int: The number of modified documents.
    """
    result = existing_collection.update_one(query, {'$set': update_data})
    return result.modified_count

#____________________________________________________________
def insert_camera_info(CameraName, SourceType, Source):
    """
    Insert camera information into the database.

    Args:
        CameraName (str): Name of the camera.
        SourceType (str): Type of the source (e.g., WEBCAM, RTSP, URL).
        Source (str): Source of the camera feed (port number for WEBCAM or URL for RTSP/URL).

    Returns:
        pymongo.results.InsertOneResult: The result of the insertion operation.
    """
    existing_collection = db['CameraInfo']
    current_time_utc = datetime.utcnow()
    egypt_tz = pytz.timezone('Africa/Cairo')
    current_time_egypt = current_time_utc.replace(tzinfo=pytz.utc).astimezone(egypt_tz)
    date_str = current_time_egypt.strftime("%Y-%m-%d")

    data = {
        'Camera Name': CameraName,
        'Source Type': SourceType,
    }

    if SourceType == 'WEBCAM':
        data['Port'] = int(Source)
    elif SourceType in ['RTSP', 'URL']:
        data['Link'] = Source

    data['Status'] = 'OFF'
    data['Insertion Timestamp'] = current_time_egypt
    data['Insertion Date'] = date_str

    query = {'Camera Name': CameraName}
    if check_existing_document(existing_collection, query):
        print('This Camera Name Already Exists')
        insetion_status = {'Inserted' : False}
        print(insetion_status)
        return  insetion_status
    else:
        inserted_document = existing_collection.insert_one(data)
        print('Inserted Successfully with ID:', inserted_document.inserted_id)
        insetion_status = {'Inserted' : True}
        print(insetion_status)
        return insetion_status

#____________________________________________________________
def insert_model_info(CameraName, ModelName, Label, FramePath):
    """
    Insert model information into the appropriate collection based on the ModelName.

    Args:
        CameraName (str): Name of the camera.
        ModelName (str): Name of the model.
        Label: Label or count associated with the model result.
        FramePath (str): Path of the processed image.

    Returns:
        pymongo.results.InsertOneResult: The result of the insertion operation.
    """

    existing_collection = {}
    if ModelName == 'violence':
        existing_collection = db['ModelViolenceData']
    elif ModelName == 'vehicle':
        existing_collection = db['ModelVehicleData']
    elif ModelName == 'crowdedDensity':
        existing_collection = db['ModelDensityData']
    elif ModelName == 'crossingBorder':
        existing_collection = db['ModelCountingData']
    elif ModelName == 'crowded':
        existing_collection = db['ModelCrowdedData']
    elif ModelName == 'gender':
        existing_collection = db['ModelGenderData']
    elif ModelName == 'clothes color':
        existing_collection = db['ModelClothesColorData']
    elif ModelName == 'Age':
        existing_collection = db['ModelAgeData']    
    elif ModelName == 'enter exit counting':
        existing_collection = db['ModelEnterEXitCountingCollection']

        
    current_time_utc = datetime.utcnow()
    egypt_tz = pytz.timezone('Africa/Cairo')
    current_time_egypt = current_time_utc.replace(tzinfo=pytz.utc).astimezone(egypt_tz)
    date_str = current_time_egypt.strftime("%Y-%m-%d")

    data = {
        'Model Name': ModelName,
    }

    query = {'Camera Name': CameraName}
    camera_collection = db['CameraInfo']
    if check_existing_document(camera_collection, query):
        print('Camera Found')
        update_existing_document(camera_collection, query, {'Status': 'ON'})
        camera_data = find_existing_document(camera_collection, query)
        data['Camera Info'] = camera_data
    else:
        print('Camera Not Added in Camera Collection')
        return 'Camera Not Added in Camera Collection'

    if isinstance(Label, int) or ModelName in ['vehicle', 'crowdedDensity', 'crossingBorder', 'crowded' , 'enter exit counting']:
        data['Count'] = Label
    elif isinstance(Label, str) or ModelName not in ['vehicle', 'crowdedDensity', 'crossingBorder', 'crowded' , ]:
        data['Label'] = Label

    data['Frame Path'] = FramePath
    data['Timestamp'] = current_time_egypt
    data['Date'] = date_str
    inserted_document = ''
    inserted_document = existing_collection.insert_one(data)
    if inserted_document != '':
        print(f'Inserted Successfully with ID in {ModelName} Collection: {inserted_document.inserted_id}')
        insetion_status = {'Inserted' : True}
        print(insetion_status)
        return  insetion_status
    else:
        insetion_status = {'Inserted' : False}
        print('Cannot Insertion')
        insetion_status = {'Inserted' : True}
        print(insetion_status)
        return insetion_status    

#____________________________________________________________
def finding_camera_names():
    """
    Retrieve all camera names from the database.

    Returns:
        list: List of camera names.
    """
    existing_collection = db['CameraInfo']
    cursor = existing_collection.find({})
    camera_names = [document['Camera Name'] for document in cursor]
    return camera_names

#____________________________________________________________
def date_filter_aggerigates_df(CameraName, ModelName, day, month, year):
    """
    Filter data by date and get the average count in the form of time range.

    Args:
        CameraName (str): Name of the camera.
        ModelName (str): Name of the model.
        day (int): Day component of the date.
        month (int): Month component of the date.
        year (int): Year component of the date.

    Returns:
        pandas.DataFrame: DataFrame containing time range and average count.
    """
    # Ensure month and day are zero-padded if less than 10
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)

    # Construct target date string
    TargetDate = f"{year}-{month_str}-{day_str}"

    # Map ModelName to corresponding collection
    model_collection_mapping = {
        'violence': 'ModelViolenceData',
        'vehicle': 'ModelVehicleData',
        'crowdedDensity': 'ModelDensityData',
        'crossingBorder': 'ModelCountingData',
        'crowded': 'ModelCrowdedData',
        'Gender': 'ModelGenderData',
        'clothes color': 'ModelClothesColorData'
        
    }



    # Get the collection based on ModelName
    collection_name = model_collection_mapping.get(ModelName)
    if not collection_name:
        return pd.DataFrame()  # Return empty DataFrame if ModelName is invalid

    existing_collection = db[collection_name]

    # query = {'Camera Info.Camera Name': CameraName}

    query = {'Camera Info.Camera Name': CameraName,'Date': TargetDate}
    if check_existing_document(existing_collection, query):
        print(f'{CameraName} Camera Found in {ModelName} Collection')
        pipeline = [
            {"$match": query},
            {"$group": {"_id": {"$hour": "$Timestamp"}, "count": {"$avg": "$Count"}}},
            {"$project": {"Hour": "$_id", "Count Average": {"$ceil": "$count"}, "_id": 0}},
            {"$sort": {"Hour": 1}}
        ]


        result = list(existing_collection.aggregate(pipeline))

        if result:
            data = []

            for item in result:
                hour = item['Hour']
                average_count = item['Count Average']
                am_pm = "PM" if (hour + 2) >= 12 else "AM"
                formatted_hour = (hour + 2) if (hour + 2) <= 12 else (hour + 2) - 12
                time_range = f"{formatted_hour} {am_pm} - {formatted_hour + 1} {am_pm}"
                data.append({'Time Range': time_range, 'Count Average': average_count})
            
            return pd.DataFrame(data)
        else:
            return pd.DataFrame()
    else:
        return pd.DataFrame()
#print(date_filter_aggerigates_df('Ro7Elsharq', 'crowded','8','2','2024'))    
        
    
#________________________________________________________________________
def months_filter_aggerigates_df(CameraName, ModelName, month, year):
    """
    Filter data by date and get the average count per each day in a month for the specified camera.

    Args:
        CameraName (str): Name of the camera.
        ModelName (str): Name of the model.
        month (string): Month component of the date.
        year (string): Year component of the date.

    Returns:
        pandas.DataFrame: DataFrame containing date and average count for each day in the month.
    """
    month_int = int(month)
    year = int(year)

    model_collection_mapping = {
        'vehicle': 'ModelVehicleData',
        'crowdedDensity': 'ModelDensityData',
        'crossingBorder': 'ModelCountingData',
        'crowded': 'ModelCrowdedData',
    }

    # Get the collection based on ModelName
    collection_name = model_collection_mapping.get(ModelName)
    if not collection_name:
        return pd.DataFrame()  # Return empty DataFrame if ModelName is invalid

    existing_collection = db[collection_name]

    # Ensure month and year are zero-padded if less than 10
    month_str = str(month_int).zfill(2)
    year_str = str(year)

    # Get the number of days in the specified month
    days_in_month = calendar.monthrange(year, month_int)[1]

    data = []

    for day in range(1, days_in_month + 1):
        day_str = str(day).zfill(2)
        TargetDate = f"{year_str}-{month_str}-{day_str}"

        query = {'Camera Info.Camera Name': CameraName, 'Date': TargetDate}

        if check_existing_document(existing_collection, query):
            pipeline = [
                {"$match": query},
                {"$group": {"_id": "$Date", "count": {"$avg": "$Count"}}},
                {"$project": {"Date": "$_id", "Count Average": {"$ceil": "$count"}, "_id": 0}}
            ]

            result = list(existing_collection.aggregate(pipeline))

            if result:
                for item in result:
                    data.append({'Days': item['Date'], 'Count Average': item['Count Average']})
            else:
                # If no data available for this day, add a record with zero count
                data.append({'Days': TargetDate, 'Count Average': 0})
    df = pd.DataFrame(data)
    df['Camera Name'] =CameraName
    df['Model'] = ModelName
    return df
#print(months_filter_aggerigates_df('Ro7Elsharq', 'crowded', '2','2024'))
#________________________________________________________________________

def year_filter_aggerigates_df(CameraName, ModelName, year):
    """
    Filter data by date and get the average count per each month in a year for the specified camera.

    Args:
        CameraName (str): Name of the camera.
        ModelName (str): Name of the model.
        year (string): Year component of the date.

    Returns:
        pandas.DataFrame: DataFrame containing month and average count for each month in the year.
    """
    year_int = int(year)

    model_collection_mapping = {
        'vehicle': 'ModelVehicleData',
        'crowdedDensity': 'ModelDensityData',
        'crossingBorder': 'ModelCountingData',
        'crowded': 'ModelCrowdedData',
    }

    # Get the collection based on ModelName
    collection_name = model_collection_mapping.get(ModelName)
    if not collection_name:
        return pd.DataFrame()  # Return empty DataFrame if ModelName is invalid

    existing_collection = db[collection_name]

    # Ensure year is zero-padded if less than 10
    year_str = str(year_int)

    # Get the number of months in a year
    months_in_year = 12

    data = []

    for month in range(1, months_in_year + 1):
        month_str = str(month).zfill(2)

        query = {'Camera Info.Camera Name': CameraName, 'Date': {'$regex': f'^{year_str}-{month_str}-'}}

        if check_existing_document(existing_collection, query):
            pipeline = [
                {"$match": query},
                {"$group": {"_id": {"$substr": ["$Date", 0, 10]}, "count": {"$avg": "$Count"}}},
                {"$project": {"Date": "$_id", "Count Average": "$count", "_id": 0}}
            ]

            result = list(existing_collection.aggregate(pipeline))

            if result:
                for item in result:
                    data.append({'Month': item['Date'][:7], 'Count Average': item['Count Average']})
            else:
                # If no data available for this month, add a record with zero count
                data.append({'Month': f"{year_str}-{month_str}", 'Count Average': 0})

    # Create DataFrame outside the loop
    df = pd.DataFrame(data)
    try :
        average_per_month = df.groupby(df['Month']).mean()
        average_per_month.reset_index(inplace=True)  # Reset index
        average_per_month['Count Average'] = average_per_month['Count Average'].apply(math.ceil).astype(float)
        average_per_month['Camera Name'] = CameraName
        average_per_month['Model'] = ModelName
        return average_per_month

    except :
            df['Camera Name'] = CameraName
            df['Model'] = ModelName        
            return df
            

#x= year_filter_aggerigates_df('Ro7Elsharq', 'crowded', '2024')
#print(x)
#____________________________________________________________
def average_camera_count(CameraName, ModelName):
    """
    Calculate the average count of a model for a specific camera.

    Args:
        CameraName (str): Name of the camera.
        ModelName (str): Name of the model.

    Returns:
        dict: Dictionary containing camera name, model name, and average count.
    """
    if ModelName == 'violence':
        existing_collection = db['ModelViolenceData']
    elif ModelName == 'vehicle':
        existing_collection = db['ModelVehicleData']
    elif ModelName == 'crowdedDensity':
        existing_collection = db['ModelDensityData']
    elif ModelName == 'crossingBorder':
        existing_collection = db['ModelCountingData']
    elif ModelName == 'crowded':
        existing_collection = db['ModelCrowdedData']
    elif ModelName == 'Gender':
        existing_collection = db['ModelGenderData']

    query = {'Camera Info.Camera Name': CameraName}

    if check_existing_document(existing_collection, query):
        print(f'{CameraName} Camera Found in {ModelName} Collection')
        pipeline = [
            {
                "$match": query
            },
            {
                "$group": {
                    "_id": "$Camera Info.Camera Name",
                    "count": {"$avg": "$Count"}
                }
            },
            {
                "$project": {
                    "Camera Name": "$_id",
                    "Count Average": "$count",
                    "_id": 0
                }
            },
            {
                "$sort": {"Camera Name": 1}
            }
        ]

        try:
            result = list(existing_collection.aggregate(pipeline))

            if result:
                result = result[0]
                result['Count Average'] = math.ceil(result['Count Average'])
                result['Model'] = ModelName
                print(result)
                return result

        except:
            nulldic = {
                'Camera Name': CameraName,
                'Model': ModelName,
                'Count Average': 'Cannot Calculate'
            }
            return nulldic
    else:
        nulldic = {
            'Camera Name': CameraName,
            'Model': ModelName,
            'Count Average': 'Not Available Data'
        }
        return nulldic

#____________________________________________________________
def postcam_geteachmodelstat(CameraName):
    """
    Calculate the average count of all models for a specific camera.

    Args:
        CameraName (str): Name of the camera.

    Returns:
        pandas.DataFrame: DataFrame containing camera name, model name, and average count.
    """

    model_collection_mapping = {
        'crowdedDensity': 'ModelDensityData',
        'crossingBorder': 'ModelCountingData',
        'crowded': 'ModelCrowdedData',
        'vehicle': 'ModelVehicleData',
        
    }

    average_counts = []

    for model_name, collection_name in model_collection_mapping.items():
        existing_collection = db[collection_name]

        query = {'Camera Info.Camera Name': CameraName}

        if check_existing_document(existing_collection, query):
            pipeline = [
                {
                    "$match": query
                },
                {
                    "$group": {
                        "_id": "$Camera Info.Camera Name",
                        "count": {"$avg": "$Count"}
                    }
                },
                {
                    "$project": {
                        "Camera Name": "$_id",
                        "Count Average": "$count",
                        "Model": model_name,
                        "_id": 0
                    }
                }
            ]

            try:
                result = list(existing_collection.aggregate(pipeline))

                if result:
                    result = result[0]
                    result['Count Average'] = math.ceil(result['Count Average'])
                    average_counts.append(result)
            except:
                pass

    # Create DataFrame from the collected data
    df = pd.DataFrame(average_counts)

    return df

#____________________________________________________________
# all models in all time
def postcam_getallmodelsStat(CameraName):
    """
    Calculate the average count of all models for a specific camera.

    Args:
        CameraName (str): Name of the camera.

    Returns:
        pandas.DataFrame: DataFrame containing camera name, model name, and average count.
    """

    model_collection_mapping = {
        'crowdedDensity': 'ModelDensityData',
        'crossingBorder': 'ModelCountingData',
        'crowded': 'ModelCrowdedData',
    }
    try :
        average_counts = []

        for model_name, collection_name in model_collection_mapping.items():
            existing_collection = db[collection_name]

            query = {'Camera Info.Camera Name': CameraName}

            if check_existing_document(existing_collection, query):
                pipeline = [
                    {
                        "$match": query
                    },
                    {
                        "$group": {
                            "_id": "$Camera Info.Camera Name",
                            "count": {"$avg": "$Count"}
                        }
                    },
                    {
                        "$project": {
                            "Camera Name": "$_id",
                            "Count Average": "$count",
                            "Model": model_name,
                            "_id": 0
                        }
                    }
                ]

                try:
                    result = list(existing_collection.aggregate(pipeline))

                    if result:
                        result = result[0]
                        result['Count Average'] = math.ceil(result['Count Average'])
                        average_counts.append(result)
                        
                        # Break the loop if CameraName is found
                        break
                except:
                    pass

        # Create DataFrame from the collected data
        df = pd.DataFrame([average_counts[0]])
        return average_counts[0]
    
    except :
        dictionary = {'Camera Name' :CameraName , 'Count Average' :0 , 'Model' : 'Not Found'  }
        return dictionary


#____________________________________________________________
# each model in specific day
def postcam_geteachmodelperH(CameraName, day, month, year):
    """
    Filter data by date and get the average count in the form of time range for all models applied to a specific camera.

    Args:
        CameraName (str): Name of the camera.
        day (int): Day component of the date.
        month (int): Month component of the date.
        year (int): Year component of the date.

    Returns:
        pandas.DataFrame: DataFrame containing time range, model name, and average count.
    """
    # Ensure month and day are zero-padded if less than 10
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)

    # Construct target date string
    TargetDate = f"{year}-{month_str}-{day_str}"

    # Map ModelName to corresponding collection
    model_collection_mapping = {
        'vehicle': 'ModelVehicleData',
        'crowdedDensity': 'ModelDensityData',
        'crossingBorder': 'ModelCountingData',
        'crowded': 'ModelCrowdedData',
    }

    data = []

    for ModelName, collection_name in model_collection_mapping.items():
        existing_collection = db[collection_name]

        query = {'Camera Info.Camera Name': CameraName, 'Date': TargetDate}
        
        if check_existing_document(existing_collection, query):
            print(f'{CameraName} Camera Found in {ModelName} Collection')
            pipeline = [
                {"$match": query},
                {"$group": {"_id": {"$hour": "$Timestamp"}, "count": {"$avg": "$Count"}}},
                {"$project": {"Hour": "$_id", "Count Average": {"$ceil": "$count"}, "_id": 0}},
                {"$sort": {"Hour": 1}}
            ]

            result = list(existing_collection.aggregate(pipeline))

            if result:
                for item in result:
                    hour = item['Hour']
                    average_count = item['Count Average']
                    am_pm = "PM" if (hour + 2) >= 12 else "AM"
                    formatted_hour = (hour + 2) if (hour + 2) <= 12 else (hour + 2) - 12
                    time_range = f"{formatted_hour} {am_pm} - {formatted_hour + 1} {am_pm}"
                    data.append({'Model': ModelName, 'Time Range': time_range, 'Count Average': average_count})

    return pd.DataFrame(data)


#____________________________________________________________
# all models in specific day
def postcam_getallmodelsperH(CameraName, day, month, year):
    """
    Filter data by date and get the average count in the form of time range for all models applied to a specific camera.

    Args:
        CameraName (str): Name of the camera.
        day (int): Day component of the date.
        month (int): Month component of the date.
        year (int): Year component of the date.

    Returns:
        pandas.DataFrame: DataFrame containing time range, model name, and average count.
    """
    # Ensure month and day are zero-padded if less than 10
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)

    # Construct target date string
    TargetDate = f"{year}-{month_str}-{day_str}"

    # Map ModelName to corresponding collection
    model_collection_mapping = {
        'crossingBorder': 'ModelCountingData',        
        'crowdedDensity': 'ModelDensityData',
        'crowded': 'ModelCrowdedData',
    }

    data = []
    try:
        for ModelName, collection_name in model_collection_mapping.items():
            existing_collection = db[collection_name]

            query = {'Camera Info.Camera Name': CameraName, 'Date': TargetDate}

            if check_existing_document(existing_collection, query):
                print(f'{CameraName} Camera Found in {ModelName} Collection')
                pipeline = [
                    {"$match": query},
                    {"$group": {"_id": {"$hour": "$Timestamp"}, "count": {"$avg": "$Count"}}},
                    {"$project": {"Hour": "$_id", "Count Average": {"$ceil": "$count"}, "_id": 0}},
                    {"$sort": {"Hour": 1}}
                ]

                result = list(existing_collection.aggregate(pipeline))
                if result:
                    for item in result:
                        hour = item['Hour'] + 2
                        formatted_hour = hour if hour <= 12 else hour - 12
                        average_count = item['Count Average']
                        am_pm = "PM" if hour >= 12 else "AM"
                        time_range = f"{formatted_hour} {am_pm} - {formatted_hour + 1 if formatted_hour < 12 else 1} {am_pm}"
                        if time_range == '12 PM - 13 PM' :
                                time_range = time_range.replace(time_range,'12 PM - 1 PM')                           
                        data.append({'Model': ModelName, 'Time Range': time_range, 'Count Average': average_count})


                    for hour in range(24):
                        am_pm = "PM" if (hour) >= 12 else "AM"
                        formatted_hour = (hour) if (hour) <= 12 else (hour) - 12
                        time_range = f"{formatted_hour} {am_pm} - {formatted_hour + 1} {am_pm}"


                        data.append({'Model': ModelName, 'Time Range': time_range, 'Count Average': 0})

                    break

        if len(data) == 0:
            dictionary = {'Camera Name': CameraName, 'Time Range': TargetDate, 'Count Average': 0, 'Model': 'Not Found'}
            return dictionary
        else:
            df = pd.DataFrame(data)
            df['Camera Name'] = CameraName

            # Define the order of time ranges
            df = df.sort_values(by=['Time Range', 'Count Average'], ascending=[True, False]).drop_duplicates(subset='Time Range')
            # Define the order of time ranges
            time_range_order = [f"{i} AM - {i + 1} AM" if i != 12 else f"{i} AM - {i + 1} PM" for i in range(12)]
            time_range_order.extend([f"{i} PM - {i + 1} PM" if i != 12 else f"{i} PM - {i + 1} AM" for i in range(12)])
            ind =  time_range_order.index('0 PM - 1 PM')
            time_range_order[ind] = '12 PM - 1 PM'

            # Convert 'Time Range' column to categorical with predefined order
            df['Time Range'] = pd.Categorical(df['Time Range'], categories=time_range_order, ordered=True)

            # Sort DataFrame by 'Time Range'
            df.sort_values(by='Time Range', inplace=True)
            df.dropna(inplace=True)
            return df

    except Exception as e:
        print(e)
        dictionary = {'Camera Name': CameraName, 'Time Range': 'Null', 'Count Average': 0, 'Models': 'Not Found'}
        return dictionary
#print(postcam_getallmodelsperH('aslom',19,2,2024))
    
#____________________________________________________________________________    
def postcam_geteachmodelperM(CameraName, month, year):
    """
    Filter data by date and get the average count per each day in a month for the specified camera.

    Args:
        CameraName (str): Name of the camera.
        month (string): Month component of the date.
        year (string): Year component of the date.

    Returns:
        pandas.DataFrame: DataFrame containing date and average count for each day in the month.
    """
    month_int = int(month)
    year = int(year)

    model_collection_mapping = {
        'vehicle': 'ModelVehicleData',
        'crowdedDensity': 'ModelDensityData',
        'crossingBorder': 'ModelCountingData',
        'crowded': 'ModelCrowdedData',
    }


    # Ensure month and year are zero-padded if less than 10
    month_str = str(month_int).zfill(2)
    year_str = str(year)

    # Get the number of days in the specified month
    days_in_month = calendar.monthrange(year, month_int)[1]

    data = []

    for model_name, collection_name in model_collection_mapping.items():
        existing_collection = db[collection_name]

        for day in range(1, days_in_month + 1):
            day_str = str(day).zfill(2)
            TargetDate = f"{year_str}-{month_str}-{day_str}"

            query = {'Camera Info.Camera Name': CameraName, 'Date': TargetDate}

            if check_existing_document(existing_collection, query):
                pipeline = [
                    {"$match": query},
                    {"$group": {"_id": "$Date", "count": {"$avg": "$Count"}}},
                    {"$project": {"Date": "$_id", "Count Average": {"$ceil": "$count"}, "_id": 0}}
                ]

                result = list(existing_collection.aggregate(pipeline))

                if result:
                    for item in result:
                        data.append({'Model': model_name, 'Time Range': item['Date'], 'Count Average': item['Count Average']})
                else:
                    # If no data available for this day, add a record with zero count
                    data.append({'Model': model_name, 'Time Range': TargetDate, 'Count Average': 0})
            else:
                    # If no data available for this day, add a record with zero count
                    data.append({'Model': model_name, 'Time Range': TargetDate, 'Count Average': 0})                    
    if len(data) == 0 :
            dictionary = {'Camera Name' :CameraName , 'Count Average' :0 , 'Model' : 'Not Found'  }
            return dictionary
    else :                
        return pd.DataFrame(data)

#print(postcam_geteachmodelperM('Ro7Elsharq','02','2024'))
    
#____________________________________________________________________________
def postcam_getallmodelsperM(CameraName, month, year):
    """
    Filter data by date and get the average count per each day in a month for the specified camera.

    Args:
        CameraName (str): Name of the camera.
        month (string): Month component of the date.
        year (string): Year component of the date.

    Returns:
        pandas.DataFrame: DataFrame containing date and average count for each day in the month.
    """
    month_int = int(month)
    year = int(year)

    model_collection_mapping = {
        'crossingBorder': 'ModelCountingData',
        'crowded': 'ModelCrowdedData',
        'crowdedDensity': 'ModelDensityData',
    }

    # Ensure month and year are zero-padded if less than 10
    month_str = str(month_int).zfill(2)
    year_str = str(year)

    # Get the number of days in the specified month
    days_in_month = calendar.monthrange(year, month_int)[1]

    data = []

    data_found = False  # Flag to track whether data has been found
    camera_found = False  # Flag to track whether camera is found

    try:
        for model_name, collection_name in model_collection_mapping.items():
            existing_collection = db[collection_name]
            if data_found:  # If data is already found, break the loop
                    break            
            for day in range(1, days_in_month + 1):
                
                day_str = str(day).zfill(2)
                TargetDate = f"{year_str}-{month_str}-{day_str}"

                query = {'Camera Info.Camera Name': CameraName, 'Date': TargetDate}

                if check_existing_document(existing_collection, query):
                    camera_found = True  # Set the flag to True indicating camera is found
                    
                    pipeline = [
                        {"$match": query},
                        {"$group": {"_id": "$Date", "count": {"$avg": "$Count"}}},
                        {"$project": {"Date": "$_id", "Count Average": {"$ceil": "$count"}, "_id": 0}}
                    ]

                    result = list(existing_collection.aggregate(pipeline))

                    if result:
                        for item in result:
                            data.append({'Model': model_name, 'Time Range': item['Date'],
                                         'Camera Name': CameraName, 'Count Average': item['Count Average']})
                        data_found = True  # Set the flag to True indicating data is found
                    else:
                        # If no data available for this day, add a record with zero count
                        data.append({'Model': model_name, 'Time Range': TargetDate, 'Count Average': 0, 'Camera Name': CameraName})
                else:
                        # If no data available for this day, add a record with zero count
                        data.append({'Model': model_name, 'Time Range': TargetDate, 'Count Average': 0, 'Camera Name': CameraName})                        

        if not camera_found:
            dictionary = {'Camera Name' :CameraName , 'Count Average' :0 , 'Model' : 'Not Found'  }
            return dictionary
        
        else:
            return pd.DataFrame(data)
    
    except Exception as e:
        print(e)
        return f"Error occurred: {str(e)}"
    
# print(postcam_getallmodelsperM('io','02','2024'))    
    
#____________________________________________________________________________
def postcam_geteachmodelperY(CameraName, year):
    """
    Filter data by date and get the average count per each month in a year for the specified camera.

    Args:
        CameraName (str): Name of the camera.
        year (string): Year component of the date.

    Returns:
        pandas.DataFrame: DataFrame containing month, average count, camera name, and model for each month in the year.
    """
    year_int = int(year)

    model_collection_mapping = {
        'vehicle': 'ModelVehicleData',
        'crowded': 'ModelCrowdedData',
        'crowdedDensity': 'ModelDensityData',
        'crossingBorder': 'ModelCountingData',
    }

    all_dfs = []  # List to store all DataFrames

    # Iterate over the model collection mapping
    for modelname, collection_name in model_collection_mapping.items():
        existing_collection = db[collection_name] 

        # Ensure year is zero-padded if less than 10
        year_str = str(year_int)

        # Get the number of months in a year
        months_in_year = 12

        data = []

        for month in range(1, months_in_year + 1):
            month_str = str(month).zfill(2)

            query = {'Camera Info.Camera Name': CameraName, 'Date': {'$regex': f'^{year_str}-{month_str}-'}}

            if check_existing_document(existing_collection, query):
                pipeline = [
                    {"$match": query},
                    {"$group": {"_id": {"$substr": ["$Date", 0, 10]}, "count": {"$avg": "$Count"}}},
                    {"$project": {"Date": "$_id", "Count Average": "$count", "_id": 0}}
                ]

                result = list(existing_collection.aggregate(pipeline))

                if result:
                    for item in result:
                        data.append({'Month': calendar.month_name[int(month_str)], 'Count Average': item['Count Average']})
                else:
                    # If no data available for this month, add a record with zero count
                    data.append({'Month':calendar.month_name[int(month_str)] , 'Count Average': 0})
            else:
                    # If no data available for this month, add a record with zero count
                    data.append({'Month':calendar.month_name[int(month_str)] , 'Count Average': 0})                    

        # Create DataFrame
        df = pd.DataFrame(data)
        try:
            average_per_month = df.groupby(df['Month']).mean()
            average_per_month.reset_index(inplace=True)  # Reset index
            average_per_month['Count Average'] = average_per_month['Count Average'].apply(math.ceil).astype(float)
            average_per_month['Camera Name'] = CameraName
            average_per_month['Model'] = modelname
            average_per_month.rename({'Month' : 'Time Range'} , axis=1,inplace=True)            
            all_dfs.append(average_per_month)  # Append DataFrame to the list
        except:
            df['Camera Name'] = CameraName
            df['Model'] = modelname
            all_dfs.append(df)  # Append DataFrame to the list

    # Concatenate all DataFrames
    final_df = pd.concat(all_dfs)
    if len(data) == 0 :
            dictionary = {'Camera Name' :CameraName , 'Count Average' :0 , 'Model' : 'Not Found'  }
            return dictionary
    else :
        # Sort by month order
        month_order = list(calendar.month_name)[1:]  # Remove the first element which is an empty string
        final_df['Month'] = pd.Categorical(final_df['Time Range'], categories=month_order, ordered=True)
        final_df.sort_values('Month', inplace=True)
        final_df.drop(columns='Month', inplace=True)  # Remove temporary 'Month' column used for sorting
        return final_df                             

#print(postcam_geteachmodelperY('Ro7Elsharq',2024))    
    
#____________________________________________________________________________    
def postcam_getallmodelsperY(CameraName, year):
    """
    Filter data by date and get the average count per each month in a year for the specified camera.

    Args:
        CameraName (str): Name of the camera.
        year (string): Year component of the date.

    Returns:
        pandas.DataFrame: DataFrame containing month, average count, camera name, and model for each month in the year.
    """
    year_int = int(year)

    model_collection_mapping = {
        'crowded': 'ModelCrowdedData',
        'crowdedDensity': 'ModelDensityData',
        'crossingBorder': 'ModelCountingData',
    }

    all_dfs = []  # List to store all DataFrames
    found = False
    # Iterate over the model collection mapping
    for modelname, collection_name in model_collection_mapping.items():
        if found :
            break
        existing_collection = db[collection_name] 

        # Ensure year is zero-padded if less than 10
        year_str = str(year_int)

        # Get the number of months in a year
        months_in_year = 12

        data = []

        for month in range(1, months_in_year + 1):
            month_str = str(month).zfill(2)

            query = {'Camera Info.Camera Name': CameraName, 'Date': {'$regex': f'^{year_str}-{month_str}-'}}

            if check_existing_document(existing_collection, query):
                print(CameraName , 'is in ', collection_name)
                pipeline = [
                    {"$match": query},
                    {"$group": {"_id": {"$substr": ["$Date", 0, 10]}, "count": {"$avg": "$Count"}}},
                    {"$project": {"Date": "$_id", "Count Average": "$count", "_id": 0}}
                ]

                result = list(existing_collection.aggregate(pipeline))

                if result:
                    for item in result:
                        data.append({'Month': calendar.month_name[int(month_str)], 'Count Average': item['Count Average']})
                        found = True                        
                else:
                    # If no data available for this month, add a record with zero count
                    data.append({'Month':calendar.month_name[int(month_str)] , 'Count Average': 0})
            else : 
                    data.append({'Month':calendar.month_name[int(month_str)] , 'Count Average': 0})

        # Create DataFrame
        df = pd.DataFrame(data)
        try:
            average_per_month = df.groupby(df['Month']).mean()
            average_per_month.reset_index(inplace=True)  # Reset index
            average_per_month['Count Average'] = average_per_month['Count Average'].apply(math.ceil).astype(float)
            average_per_month['Camera Name'] = CameraName
            average_per_month['Model'] = modelname
            average_per_month.rename({'Month' : 'Time Range'} , axis=1,inplace=True)            
            all_dfs.append(average_per_month)  # Append DataFrame to the list
        except:
            df['Camera Name'] = CameraName
            df['Model'] = modelname
            all_dfs.append(df)  # Append DataFrame to the list

    # Concatenate all DataFrames
    final_df = pd.concat(all_dfs)
    if len(data) == 0 :
            dictionary = {'Camera Name' :CameraName , 'Count Average' :0 , 'Model' : 'Not Found'  }
            return dictionary
    else :             
        # Sort by month order
        month_order = list(calendar.month_name)[1:]  # Remove the first element which is an empty string
        final_df['Month'] = pd.Categorical(final_df['Time Range'], categories=month_order, ordered=True)
        final_df.sort_values('Month', inplace=True)
        final_df.drop(columns='Month', inplace=True)  # Remove temporary 'Month' column used for sorting
        count_per_model = final_df.groupby('Model')['Count Average'].sum()
        model_lists = list(count_per_model.index)
        for model in model_lists : 
            if (count_per_model[model] == 0)   :
            # Drop rows where 'Model' column contains the specific value
                final_df = final_df[final_df['Model'] != model]


    # Check if DataFrame is empty after dropping rows
        if len(final_df) == 0:
            final_df = pd.DataFrame({'Time Range': [year], 
                        'Count Average': [0], 
                        'Camera Name': [CameraName],
                        'Model': ['Not Found']})

        return final_df                
    

# x = postcam_getallmodelsperY('Ro7Elsharq',2024)
# print(x)

#____________________________________________________________________________
def all_cameras_in_model(ModelName) :
    
    model_collection_mapping = {
        'violence': 'ModelViolenceData',
        'vehicle': 'ModelVehicleData',
        'crowdedDensity': 'ModelDensityData',
        'crossingBorder': 'ModelCountingData',
        'crowded': 'ModelCrowdedData',
        'gender': 'ModelGenderData',
        'clothes color': 'ModelClothesColorData'

          }

    # Get the collection based on ModelName
    collection_name = model_collection_mapping.get(ModelName)
    
    if not collection_name:
        dictionary = {'Model' : ModelName , 'Error' :' Don\'t Collection' }
        return dictionary
    
    existing_collection = db[collection_name]
    print(f'Founding Data for {ModelName} Model')
    distinct_camera_names = existing_collection.distinct('Camera Info.Camera Name')

    dictionary = {'Model' : ModelName ,'Camera Names' :distinct_camera_names }
    return dictionary

#____________________________________________________________________________
def all_models_in_camera(CameraName) :
    
    model_collection_mapping = {
        'violence': 'ModelViolenceData',
        'vehicle': 'ModelVehicleData',
        'crowdedDensity': 'ModelDensityData',
        'crossingBorder': 'ModelCountingData',
        'crowded': 'ModelCrowdedData',
        'gender': 'ModelGenderData',
        'clothes color': 'ModelClothesColorData'
}
    
    models_list = []
    
    for model_name, collection_name in model_collection_mapping.items():
        existing_collection = db[collection_name] 
        query = {'Camera Info.Camera Name': CameraName}
        if check_existing_document(existing_collection, query) :
            models_list.append(model_name)     

    if len(models_list) == 0 :
        dictionary = {'Camera Name' : CameraName , 'Models' : 'Not Found'}
        return dictionary 
    
    else :     
        dictionary = {'Camera Name' : CameraName , 'Models' : models_list}
        return dictionary

def all_camera_info(CameraName) :
    camera_info_collection = db['CameraInfo'] 
    query = {'Camera Name': CameraName}    
    if check_existing_document(camera_info_collection, query) :
        data = find_existing_document(camera_info_collection, query)
        data.pop('_id', None)

        model_collection_mapping = {
            'violence': 'ModelViolenceData',
            'vehicle': 'ModelVehicleData',
            'crowdedDensity': 'ModelDensityData',
            'crossingBorder': 'ModelCountingData',
            'crowded': 'ModelCrowdedData',
            'gender': 'ModelGenderData',
            'clothes color': 'ModelClothesColorData'
 }
        
        models_list = []
    
        for model_name, collection_name in model_collection_mapping.items():
            existing_collection = db[collection_name] 
            query = {'Camera Info.Camera Name': CameraName}
            if check_existing_document(existing_collection, query) :
                models_list.append(model_name)     
        if len(models_list) == 0 :
            dictionary = dict()
            dictionary['Camera Name'] = CameraName
            dictionary['Camera Info'] = data
            try :
                dictionary['Camera Info']['Source'] = dictionary['Camera Info'].pop('Link')
            except :
                dictionary['Camera Info']['Source'] = dictionary['Camera Info'].pop('Port')
                            
            dictionary['Models'] = []
            return dictionary
        
        else :
            dictionary = dict()
            dictionary['Camera Name'] = CameraName
            dictionary['Camera Info'] = data
            dictionary['Models'] = models_list
        
            try :
                dictionary['Camera Info']['Source'] = dictionary['Camera Info'].pop('Link')
            except :
                dictionary['Camera Info']['Source'] = dictionary['Camera Info'].pop('Port')

            return dictionary                        
    else :
        dictionary = dict()
        dictionary['Camera Name'] = CameraName
        dictionary['Camera Info'] = {}
        dictionary['Models'] = []
        return dictionary
    
    
# pprint.pprint(all_camera_info('FristCam'))
    
def clothes_filtering(CameraName, color):
    existing_collection = db['ModelClothesColorData']
    colors_list = [
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
    if color in colors_list :
        
        query = {'Camera Info.Camera Name': CameraName}
        
        if check_existing_document(existing_collection, query):
            # Search for documents containing the specified color
            query = {"Label": {"$elemMatch": {"clothes": {"$elemMatch": {"Dominant Color Name": color}}}}}
            result = existing_collection.find(query)  
            matching_documents = []
            for doc in result:
                matching_label = []
                for label in doc['Label']:
                    matching_clothes = []
                    if isinstance(label.get('clothes', []), list):  # Check if 'clothes' is a list
                        for cloth in label['clothes']:
                            if isinstance(cloth, dict) and cloth.get('Dominant Color Name') == color:
                                matching_clothes.append(cloth)
                    if matching_clothes:
                        matching_label.append({'Person ID': label['Person ID'], 'clothes': matching_clothes})
                doc_copy = doc.copy()  # Create a copy of the document
                del doc_copy['_id']  # Remove the _id field
                del doc_copy['Camera Info']['_id']
                del doc_copy['Model Name']
                doc_copy['Label'] = matching_label  # Replace 'Label' with filtered clothes
                matching_documents.append(doc_copy)
            if len(matching_documents) == 0 :
                dictionary = {'CameraName' :CameraName , 'Model' :'clothes color' , 
                            'EnteredColor' :color , 'Data' :'No Data Found'}            
                return dictionary
            
            else :
                dictionary = {'CameraName' :CameraName , 'Model' :'clothes color' ,
                            'EnteredColor' :color , 'Data' :matching_documents}            
                return dictionary
        else: 
            dictionary = {'CameraName' : CameraName ,
                        'Data' : 'Camera is not inserted' }       
            return dictionary 
    else :
        dictionary = {'CameraName' : CameraName ,
                        'Error' : 'Please Enter Color from the list' }       
        return dictionary   


# colors_list = [
#         'Red'     ,
#         'Blue'    ,
#         'Green' ,       
#         'Yellow' ,   
#         'Purple' ,
#         'Cyan' ,     
#         'Orange' ,
#         'Brown',
#         'Black' ,
#         'White' ]             
# # # Example usage:
# data = clothes_filtering('ClothesTest', 'Yellow')
# pprint.pprint(data)
    
def gender_filtering_date_aggrigates(CameraName, day, month, year):
    
    """
    Filter data by date and get the average count in the form of time range.

    Args:
        CameraName (str): Name of the camera.
        day (int): Day component of the date.
        month (int): Month component of the date.
        year (int): Year component of the date.

    Returns:
        pandas.DataFrame: DataFrame containing time range, total count for male, and total count for female.
    """
    # Ensure month and day are zero-padded if less than 10
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)

    # Construct target date string
    TargetDate = f"{year}-{month_str}-{day_str}"

    existing_collection = db['ModelGenderData']
    
    # Query to filter by camera name and date
    query = {'Camera Info.Camera Name': CameraName, 'Date': TargetDate}
    data = []
    if check_existing_document(existing_collection, query):
        print(f'{CameraName} Camera Found in Gender Collection')
        
        pipeline = [
            {"$match": query},
            {"$unwind": "$Label"},
            {"$group": {
                "_id": {
                    "Hour": {"$hour": "$Timestamp"},
                    "Gender": "$Label.Gender"
                },
                "count": {"$sum": 1}
            }},
            {"$project": {
                "Hour": "$_id.Hour",
                "Gender": "$_id.Gender",
                "Count": "$count",
                "_id": 0
            }},
            {"$sort": {"Hour": 1}}
        ]


        result = list(existing_collection.aggregate(pipeline))
        if result:
            
            for item in result:
                hour = item['Hour']
                gender = item['Gender']
                count = item['Count']
                
                # Adjusting hour for display
                am_pm = "PM" if (hour + 2) >= 12 else "AM"
                formatted_hour = (hour + 2) if (hour + 2) <= 12 else (hour + 2) - 12
                time_range = f"{formatted_hour} {am_pm} - {formatted_hour + 1} {am_pm}"
                
                data.append({
                    'Time Range': time_range,
                    'Gender': gender,
                    'Count': count,
                })

                for hour in range(24):
                        am_pm = "PM" if (hour) >= 12 else "AM"
                        formatted_hour = (hour) if (hour) <= 12 else (hour) - 12
                        time_range = f"{formatted_hour} {am_pm} - {formatted_hour + 1} {am_pm}"
                
                        data.append({
                            'Time Range': time_range,
                            'Gender': gender,
                            'Count': 0,
                        })

                        
            
            # Create DataFrame from collected data
            df = pd.DataFrame(data)
            
            # Pivot DataFrame to get male and female counts separately
            df_pivot = df.pivot_table(index='Time Range', columns='Gender', values='Count', aggfunc='sum', fill_value=0)
            
            # Reset index to make 'Time Range' a column
            df_pivot.reset_index(inplace=True)
            
            # Rename columns for clarity
            df_pivot.columns.name = None
            
            try :                                        
                    try :
                    # Add a total count column
                        df_pivot['Total Count'] = df_pivot['Female'] + df_pivot['Male']
                        df_pivot['Camera Name'] = CameraName   
                    except : 
                        df_pivot['Total Count'] = df_pivot['Female'] 
                        df_pivot['Female'] = 0                        
                        df_pivot['Camera Name'] = CameraName                   
            except :
                        df_pivot['Total Count'] = df_pivot['Male'] 
                        df_pivot['Female'] = 0                        

                        df_pivot['Camera Name'] = CameraName              
            

    if len(data) == 0:
                dictionary = {'Camera Name': CameraName, 'Time Range': TargetDate,
                              'Female' : 0 ,'Male' :0, 'Total Count': 0}
                return dictionary
    else :

            # Define the order of time ranges
            df = df_pivot.sort_values(by=['Time Range', 'Total Count'], ascending=[True, False]).drop_duplicates(subset='Time Range')
            # Define the order of time ranges
            time_range_order = [f"{i} AM - {i + 1} AM" if i != 12 else f"{i} AM - {i + 1} PM" for i in range(12)]
            time_range_order.extend([f"{i} PM - {i + 1} PM" if i != 12 else f"{i} PM - {i + 1} AM" for i in range(12)])
            ind =  time_range_order.index('0 PM - 1 PM')
            time_range_order[ind] = '12 PM - 1 PM'

            # Convert 'Time Range' column to categorical with predefined order
            df['Time Range'] = pd.Categorical(df['Time Range'], categories=time_range_order, ordered=True)

            # Sort DataFrame by 'Time Range'
            df.sort_values(by='Time Range', inplace=True)
            df.dropna(inplace=True)
            return df          
# print(gender_filtering_date_aggrigates('FristCam',9,3,2024))   
    
def gender_filtering_month_aggregates(CameraName, month, year):

    """
    Filter data by month and get the total count for male and female for each day.

    Args:
        CameraName (str): Name of the camera.
        month (int): Month component of the date.
        year (int): Year component of the date.

    Returns:
        pandas.DataFrame: DataFrame containing total count for male and female for each day.
    """
    # Ensure month is zero-padded if less than 10
    month = int(month)
    year = int(year)
    month_str = str(month).zfill(2)
    # Get the number of days in the month
    num_days = calendar.monthrange(year, month)[1]

    # Initialize an empty list to store data for all days in the month
    data = []

    # Iterate over all days in the month
    for day in range(1, num_days + 1):
        # Retrieve data for the specific day using the existing function
        daily_data = gender_filtering_date_aggrigates(CameraName, day, month, year)
        
        # If there's no data for the day, add zeros for male and female counts
        if isinstance(daily_data, dict): # If there's no data for the day
            data.append({
                'Camera Name': CameraName,
                'Time Range': f"{year}-{month_str}-{str(day).zfill(2)}",
                'Male': 0,
                'Female': 0,
                'Total Count': 0
            })
        else:
            # Calculate the total count for male and female for the day
            male_count = daily_data['Male'].sum()
            female_count = daily_data['Female'].sum()
            total_count = male_count + female_count
            
            # Add the data for the day to the list
            data.append({
                'Camera Name': CameraName,
                'Time Range': f"{year}-{month_str}-{str(day).zfill(2)}",
                'Male': male_count,
                'Female': female_count,
                'Total Count': total_count
            })

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)

    # Return the DataFrame
    return df     

# print(gender_filtering_month_aggregates('FristCam',3,2024))    

def gender_filtering_year_aggregates(CameraName, year):
    """
    Filter data by year and get the total count for male and female for each month.

    Args:
        CameraName (str): Name of the camera.
        year (int): Year component.

    Returns:
        pandas.DataFrame: DataFrame containing total count for male and female for each month.
    """
    # Initialize an empty list to store data for all months in the year
    data = []
    year = int(year)

    # Iterate over all months in the year
    for month in range(1, 13):
        # Get the number of days in the month
        num_days = calendar.monthrange(year, month)[1]

        # Initialize counters for male and female counts for the month
        male_count_month = 0
        female_count_month = 0

        # Iterate over all days in the month
        for day in range(1, num_days + 1):
            # Retrieve data for the specific day using the existing function
            daily_data = gender_filtering_date_aggrigates(CameraName, day, month, year)
            
            # If there's data for the day, sum the male and female counts
            if not isinstance(daily_data, dict): # If there's no data for the day
                male_count_month += daily_data['Male'].sum()
                female_count_month += daily_data['Female'].sum()

        # Calculate the total count for the month
        total_count_month = male_count_month + female_count_month
        
        # Add the aggregated data for the month to the list
        data.append({
            'Camera Name': CameraName,
            'Time Range': calendar.month_name[int(month)],
            'Year' : year ,
            'Male': male_count_month,
            'Female': female_count_month,
            'Total Count': total_count_month
        })

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)

    # Return the DataFrame
    return df



def VoilenceFilteringH(CameraName, day, month, year):
    existing_collection = db['ModelViolenceData']

    # Ensure month and day are zero-padded if less than 10
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)

    # Construct target date string
    TargetDate = f"{year}-{month_str}-{day_str}"

    query = {'Camera Info.Camera Name': CameraName, 'Date': TargetDate}

    # Check if documents exist for the given query
    if check_existing_document(existing_collection,query) :
        print(f'{CameraName} Camera Found in Violence Collection')
        
        # Define the aggregation pipeline with dynamic date filtering
        pipeline = [
            {
                "$match": query
            },
            {
                "$addFields": {
                    "HourOfDay": {"$hour": "$Timestamp"}
                }
            },
            {
                "$group": {
                    "_id": "$HourOfDay",
                    "documents": {"$push": "$$ROOT"}
                }
            },
            {
                "$addFields": {
                    "hours": "$_id",
                    "_id": "$$REMOVE"  # Remove the original _id field
                }
            },
                {
                "$sort": {"hours": 1}  # Sort by the "hours" field in ascending order
            }            
        ]

        # Execute the aggregation pipeline
        result = list(existing_collection.aggregate(pipeline))
        #pprint.pprint(result)
        #return result
        # Prepare data for DataFrame
        data = []
        print(f'Documents found for {CameraName} on {TargetDate}')

        for item in result:
            hour = item['hours']
            am_pm = "PM" if (hour + 2) >= 12 else "AM"
            formatted_hour = (hour + 2) if (hour + 2) <= 12 else (hour + 2) - 12
            time_range = f"{formatted_hour} {am_pm} - {formatted_hour + 1} {am_pm}"

            if time_range == '12 PM - 13 PM' :
                        time_range = time_range.replace(time_range,'12 PM - 1 PM')     
            elif time_range == '11 AM - 12 AM' :
                        time_range = time_range.replace(time_range,'11 AM - 12 PM')     

                        # Remove _id field from each document
            cleaned_documents = []
            for doc in item['documents']:
                if 'Camera Info' in doc:
                    doc['Camera Info'].pop('_id', None)
                doc.pop('_id', None)
                cleaned_documents.append(doc)

            data.append({'Time Range': time_range, 'Data': cleaned_documents, 'Camera Name': CameraName})          
            
        # Create DataFrame
        df = pd.DataFrame(data)
        return data    
    else:
        print(f'No Documents found for {CameraName} on {TargetDate}')

        dic = {'Time Range': TargetDate, 'Data': [] , 'Camera Name' :CameraName}        
        # return dic
        return [dic]
    



# test= VoilenceFilteringH('VoilenceTest',3,3,2024)
# pprint.pprint(test)    

def VoilenceFilteringM(CameraName, month, year):
    existing_collection = db['ModelViolenceData']

    month = int(month)
    year = int(year)
    # Get the number of days in the month
    num_days = calendar.monthrange(year, month)[1]

    # Prepare data to store results for each day
    monthly_data = []

    # Iterate over each day in the month
    for day in range(1, num_days + 1):
        # Ensure day and month are zero-padded if less than 10
        day_str = str(day).zfill(2)
        month_str = str(month).zfill(2)

        # Construct target date string
        TargetDate = f"{year}-{month_str}-{day_str}"

        query = {'Camera Info.Camera Name': CameraName, 'Date': TargetDate}

        # Check if documents exist for the given query
        if check_existing_document(existing_collection, query):
            print(f'{CameraName} Camera Found in Violence Collection for {TargetDate}')

            # Define the aggregation pipeline with dynamic date filtering
            pipeline = [
                {
                    "$match": query
                },
                {
                    "$group": {
                        "_id": "$Date",
                        "documents": {"$push": "$$ROOT"}
                    }
                }
            ]

            # Execute the aggregation pipeline
            result = list(existing_collection.aggregate(pipeline))
            print(f'Documents found for {CameraName} on {TargetDate}')

            # Prepare data for DataFrame
            for item in result:
                # Removing _id from each document
                cleaned_documents = []
                for doc in item['documents']:
                    if 'Camera Info' in doc:
                        doc['Camera Info'].pop('_id', None)
                        doc.pop('_id',None)
                    cleaned_documents.append(doc)

                daily_data = {
                    'Time Range': item['_id'],
                    'Data': cleaned_documents,
                    'Camera Name': CameraName
                }
                monthly_data.append(daily_data)
        else:
            print(f'No documents found for {CameraName} on {TargetDate}')

    if len(monthly_data) != 0:
        return monthly_data
    else:
        data = {
            'Time Range': f"{year}-{month_str}",
            'Camera Name': CameraName,
            'Data': []        }
        # return data
        return [data]

# result = VoilenceFilteringM('VoilenceTest', 3, 2024)
# pprint.pprint(result)


def VoilenceFilteringY(CameraName, year):
    existing_collection = db['ModelViolenceData']

    year = int(year)

    # Prepare data to store results for each month
    yearly_data = []

    # Iterate over each month in the year
    for month in range(1, 13):
        # Get the number of days in the month
        num_days = calendar.monthrange(year, month)[1]

        # Prepare data to store results for each day
        monthly_data = []

        # Get the month name
        month_name = calendar.month_name[month]

        # Construct target date string
        TargetDate = f"{month_name} {year}"

        # Iterate over each day in the month
        for day in range(1, num_days + 1):
            # Ensure day and month are zero-padded if less than 10
            day_str = str(day).zfill(2)
            month_str = str(month).zfill(2)

            # Construct target date string
            TargetDate = f"{year}-{month_str}-{day_str}"

            query = {'Camera Info.Camera Name': CameraName, 'Date': TargetDate}

            # Check if documents exist for the given query
            if check_existing_document(existing_collection, query):
                print(f'{CameraName} Camera Found in Violence Collection for {TargetDate}')

                # Define the aggregation pipeline with dynamic date filtering
                pipeline = [
                    {
                        "$match": query
                    },
                    {
                        "$group": {
                            "_id": "$Date",
                            "documents": {"$push": "$$ROOT"}
                        }
                    }
                ]

                # Execute the aggregation pipeline
                result = list(existing_collection.aggregate(pipeline))
                print(f'Documents found for {CameraName} on {TargetDate}')
                
                    # Prepare data for DataFrame
                for item in result:
                    cleaned_documents = []
                    for doc in item['documents']:
                            if 'Camera Info' in doc:
                                doc['Camera Info'].pop('_id', None)
                                doc.pop('_id', None)
                            cleaned_documents.append(doc)

                    daily_data = {
                            'Time Range': month_name,
                            'Data': cleaned_documents,
                            'Camera Name': CameraName
                        }
                    monthly_data.append(daily_data)                

                # # Prepare data for DataFrame
                # for item in result:
                #     daily_data = {
                #         'Time Range': month_name,
                #         'Data': item['documents'],
                #         'Camera Name': CameraName
                #     }
                #     monthly_data.append(daily_data)
            else:
                print(f'No documents found for {CameraName} on {TargetDate}')

        if len(monthly_data) != 0:
            yearly_data.extend(monthly_data)

    if len(yearly_data) != 0:
        return yearly_data
    else:
        data = {'Time Range': year,
                'Camera Name': CameraName,
                'Data': []}
        # return data
        return [data]
    
    
# result = VoilenceFilteringY('VoilenceTest', 2024)
# pprint.pprint(result)
    
#______________________________________________________    
def postcam_getvechile(CameraName):
    """
    Calculate the average count of all models for a specific camera.

    Args:
        CameraName (str): Name of the camera.

    Returns:
        pandas.DataFrame: DataFrame containing camera name, model name, and average count.
    """

    model_collection_mapping = {
        'vehicle': 'ModelVehicleData',
    }
    try :
        average_counts = []

        for model_name, collection_name in model_collection_mapping.items():
            existing_collection = db[collection_name]

            query = {'Camera Info.Camera Name': CameraName}

            if check_existing_document(existing_collection, query):
                pipeline = [
                    {
                        "$match": query
                    },
                    {
                        "$group": {
                            "_id": "$Camera Info.Camera Name",
                            "count": {"$avg": "$Count"}
                        }
                    },
                    {
                        "$project": {
                            "Camera Name": "$_id",
                            "Count Average": "$count",
                            "Model": model_name,
                            "_id": 0
                        }
                    }
                ]

                try:
                    result = list(existing_collection.aggregate(pipeline))

                    if result:
                        result = result[0]
                        result['Count Average'] = math.ceil(result['Count Average'])
                        average_counts.append(result)
                        
                        # Break the loop if CameraName is found
                        break
                except:
                    pass

        # Create DataFrame from the collected data
        df = pd.DataFrame([average_counts[0]])
        return average_counts[0]
    
    except :
        dictionary = {'Camera Name' :CameraName , 'Count Average' :0 , 'Model' : 'Not Found'  }
        return dictionary
    

# pprint.pprint(postcam_getvechile('VehicleTest'))

def postcam_getvechileH(CameraName, day, month, year):
    """
    Filter data by date and get the average count in the form of time range for all models applied to a specific camera.

    Args:
        CameraName (str): Name of the camera.
        day (int): Day component of the date.
        month (int): Month component of the date.
        year (int): Year component of the date.

    Returns:
        pandas.DataFrame: DataFrame containing time range, model name, and average count.
    """
    # Ensure month and day are zero-padded if less than 10
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)

    # Construct target date string
    TargetDate = f"{year}-{month_str}-{day_str}"

    # Map ModelName to corresponding collection
    model_collection_mapping = {
        'vehicle': 'ModelVehicleData',
    }

    data = []
    try:
        for ModelName, collection_name in model_collection_mapping.items():
            existing_collection = db[collection_name]

            query = {'Camera Info.Camera Name': CameraName, 'Date': TargetDate}
            if check_existing_document(existing_collection, query):
                print(f'{CameraName} Camera Found in {ModelName} Collection')
                pipeline = [
                    {"$match": query},
                    {"$group": {"_id": {"$hour": "$Timestamp"}, "count": {"$avg": "$Count"}}},
                    {"$project": {"Hour": "$_id", "Count Average": {"$ceil": "$count"}, "_id": 0}},
                    {"$sort": {"Hour": 1}}
                ]


                result = list(existing_collection.aggregate(pipeline))
                if result:
                    for item in result:
                        hour = item['Hour'] + 2
                        formatted_hour = hour if hour <= 12 else hour - 12
                        average_count = item['Count Average']
                        am_pm = "PM" if hour >= 12 else "AM"
                        time_range = f"{formatted_hour} {am_pm} - {formatted_hour + 1 if formatted_hour < 12 else 1} {am_pm}"
                        if time_range == '12 PM - 13 PM' :
                                time_range = time_range.replace(time_range,'12 PM - 1 PM')                           
                        data.append({'Model': ModelName, 'Time Range': time_range, 'Count Average': average_count})


                    for hour in range(24):
                        am_pm = "PM" if (hour) >= 12 else "AM"
                        formatted_hour = (hour) if (hour) <= 12 else (hour) - 12
                        time_range = f"{formatted_hour} {am_pm} - {formatted_hour + 1} {am_pm}"


                        data.append({'Model': ModelName, 'Time Range': time_range, 'Count Average': 0})

                    break

        if len(data) == 0:
            dictionary = {'Camera Name': CameraName, 'Time Range': TargetDate, 'Count Average': 0, 'Model': 'Not Found'}
            return dictionary
        else:
            df = pd.DataFrame(data)
            df['Camera Name'] = CameraName

            # Define the order of time ranges
            df = df.sort_values(by=['Time Range', 'Count Average'], ascending=[True, False]).drop_duplicates(subset='Time Range')
            # Define the order of time ranges
            time_range_order = [f"{i} AM - {i + 1} AM" if i != 12 else f"{i} AM - {i + 1} PM" for i in range(12)]
            time_range_order.extend([f"{i} PM - {i + 1} PM" if i != 12 else f"{i} PM - {i + 1} AM" for i in range(12)])
            ind =  time_range_order.index('0 PM - 1 PM')
            time_range_order[ind] = '12 PM - 1 PM'

            # Convert 'Time Range' column to categorical with predefined order
            df['Time Range'] = pd.Categorical(df['Time Range'], categories=time_range_order, ordered=True)

            # Sort DataFrame by 'Time Range'
            df.sort_values(by='Time Range', inplace=True)
            df.dropna(inplace=True)
            return df

    except Exception as e:
        print(e)
        dictionary = {'Camera Name': CameraName, 'Time Range': 'Null', 'Count Average': 0, 'Models': 'Not Found'}
        return dictionary
    
# print(postcam_getvechileH('FristCam',9,3,2024))
    

#____________________________________________________________________________
def postcam_getvechileM(CameraName, month, year):
    """
    Filter data by date and get the average count per each day in a month for the specified camera.

    Args:
        CameraName (str): Name of the camera.
        month (string): Month component of the date.
        year (string): Year component of the date.

    Returns:
        pandas.DataFrame: DataFrame containing date and average count for each day in the month.
    """
    month_int = int(month)
    year = int(year)

    model_collection_mapping = {
        'vehicle': 'ModelVehicleData',
    }


    # Ensure month and year are zero-padded if less than 10
    month_str = str(month_int).zfill(2)
    year_str = str(year)

    # Get the number of days in the specified month
    days_in_month = calendar.monthrange(year, month_int)[1]

    data = []

    data_found = False  # Flag to track whether data has been found
    camera_found = False  # Flag to track whether camera is found

    try:
        for model_name, collection_name in model_collection_mapping.items():
            existing_collection = db[collection_name]
            if data_found:  # If data is already found, break the loop
                    break            
            for day in range(1, days_in_month + 1):
                
                day_str = str(day).zfill(2)
                TargetDate = f"{year_str}-{month_str}-{day_str}"

                query = {'Camera Info.Camera Name': CameraName, 'Date': TargetDate}

                if check_existing_document(existing_collection, query):
                    camera_found = True  # Set the flag to True indicating camera is found
                    
                    pipeline = [
                        {"$match": query},
                        {"$group": {"_id": "$Date", "count": {"$avg": "$Count"}}},
                        {"$project": {"Date": "$_id", "Count Average": {"$ceil": "$count"}, "_id": 0}}
                    ]

                    result = list(existing_collection.aggregate(pipeline))

                    if result:
                        for item in result:
                            data.append({'Model': model_name, 'Time Range': item['Date'],
                                         'Camera Name': CameraName, 'Count Average': item['Count Average']})
                        data_found = True  # Set the flag to True indicating data is found
                    else:
                        # If no data available for this day, add a record with zero count
                        data.append({'Model': model_name, 'Time Range': TargetDate, 'Count Average': 0, 'Camera Name': CameraName})
                else:
                        # If no data available for this day, add a record with zero count
                        data.append({'Model': model_name, 'Time Range': TargetDate, 'Count Average': 0, 'Camera Name': CameraName})                        

        if not camera_found:
            dictionary = {'Camera Name' :CameraName , 'Count Average' :0 , 'Model' : 'Not Found'  }
            return dictionary
        
        else:
            return pd.DataFrame(data)
    
    except Exception as e:
        print(e)
        return f"Error occurred: {str(e)}"
    
# print(postcam_getvechileM('VehicleTest','02','2024'))    

#____________________________________________________________________________    
def postcam_getvechileY(CameraName, year):
    """
    Filter data by date and get the average count per each month in a year for the specified camera.

    Args:
        CameraName (str): Name of the camera.
        year (string): Year component of the date.

    Returns:
        pandas.DataFrame: DataFrame containing month, average count, camera name, and model for each month in the year.
    """
    year_int = int(year)

    model_collection_mapping = {
        'vehicle': 'ModelVehicleData',
    }


    all_dfs = []  # List to store all DataFrames
    found = False
    # Iterate over the model collection mapping
    for modelname, collection_name in model_collection_mapping.items():
        if found :
            break
        existing_collection = db[collection_name] 

        # Ensure year is zero-padded if less than 10
        year_str = str(year_int)

        # Get the number of months in a year
        months_in_year = 12

        data = []

        for month in range(1, months_in_year + 1):
            month_str = str(month).zfill(2)

            query = {'Camera Info.Camera Name': CameraName, 'Date': {'$regex': f'^{year_str}-{month_str}-'}}

            if check_existing_document(existing_collection, query):
                pipeline = [
                    {"$match": query},
                    {"$group": {"_id": {"$substr": ["$Date", 0, 10]}, "count": {"$avg": "$Count"}}},
                    {"$project": {"Date": "$_id", "Count Average": "$count", "_id": 0}}
                ]

                result = list(existing_collection.aggregate(pipeline))

                if result:
                    for item in result:
                        data.append({'Month': calendar.month_name[int(month_str)], 'Count Average': item['Count Average']})
                        found = True                        
                else:
                    # If no data available for this month, add a record with zero count
                    data.append({'Month':calendar.month_name[int(month_str)] , 'Count Average': 0})
            else : 
                    data.append({'Month':calendar.month_name[int(month_str)] , 'Count Average': 0})

        # Create DataFrame
        df = pd.DataFrame(data)
        try:
            average_per_month = df.groupby(df['Month']).mean()
            average_per_month.reset_index(inplace=True)  # Reset index
            average_per_month['Count Average'] = average_per_month['Count Average'].apply(math.ceil).astype(float)
            average_per_month['Camera Name'] = CameraName
            average_per_month['Model'] = modelname
            average_per_month.rename({'Month' : 'Time Range'} , axis=1,inplace=True)            
            all_dfs.append(average_per_month)  # Append DataFrame to the list
        except:
            df['Camera Name'] = CameraName
            df['Model'] = modelname
            all_dfs.append(df)  # Append DataFrame to the list

    # Concatenate all DataFrames
    final_df = pd.concat(all_dfs)
    if len(data) == 0 :
            dictionary = {'Camera Name' :CameraName , 'Count Average' :0 , 'Model' : 'Not Found'  }
            return dictionary
    else :             
        # Sort by month order
        month_order = list(calendar.month_name)[1:]  # Remove the first element which is an empty string
        final_df['Month'] = pd.Categorical(final_df['Time Range'], categories=month_order, ordered=True)
        final_df.sort_values('Month', inplace=True)
        final_df.drop(columns='Month', inplace=True)  # Remove temporary 'Month' column used for sorting
        count_per_model = final_df.groupby('Model')['Count Average'].sum()
        model_lists = list(count_per_model.index)
        for model in model_lists : 
            if (count_per_model[model] == 0)   :
            # Drop rows where 'Model' column contains the specific value
                final_df = final_df[final_df['Model'] != model]


    # Check if DataFrame is empty after dropping rows
        if len(final_df) == 0:
            final_df = pd.DataFrame({'Time Range': [year], 
                        'Count Average': [0], 
                        'Camera Name': [CameraName],
                        'Model': ['Not Found']})

        return final_df                
         
# print(postcam_getvechileY('io',2024))
    
def all_cameras_info() :
    camera_info_collection = db['CameraInfo'] 
    cameras_list  = finding_camera_names()
    cameras_info = []
    for CameraName in cameras_list :
        print('Getting Data of :' , CameraName)
        query = {'Camera Name': CameraName}    
        if check_existing_document(camera_info_collection, query) :
            data = find_existing_document(camera_info_collection, query)
            data.pop('_id', None)

            model_collection_mapping = {
                'violence': 'ModelViolenceData',
                'vehicle': 'ModelVehicleData',
                'crowdedDensity': 'ModelDensityData',
                'crossingBorder': 'ModelCountingData',
                'crowded': 'ModelCrowdedData',
                'gender': 'ModelGenderData',
                'clothes color': 'ModelClothesColorData'
    }
            
            models_list = []
        
            for model_name, collection_name in model_collection_mapping.items():
                existing_collection = db[collection_name] 
                query = {'Camera Info.Camera Name': CameraName}
                if check_existing_document(existing_collection, query) :
                    models_list.append(model_name)     
            if len(models_list) == 0 :
                dictionary = dict()
                dictionary['Camera Name'] = CameraName
                dictionary['Camera Info'] = data
            
                try :
                     if 'Link' in dictionary['Camera Info']:
                        dictionary['Camera Info']['Source'] = dictionary['Camera Info'].pop('Link')
                except :
                    if 'Port' in dictionary['Camera Info']:
                        dictionary['Camera Info']['Source'] = dictionary['Camera Info'].pop('Port')                        
                                
                dictionary['Models'] = []
                cameras_info.append(dictionary)
            
            else :
                dictionary = dict()
                dictionary['Camera Name'] = CameraName
                dictionary['Camera Info'] = data
                dictionary['Models'] = models_list
            
                try :
                    dictionary['Camera Info']['Source'] = dictionary['Camera Info'].pop('Link')
                except :
                    dictionary['Camera Info']['Source'] = dictionary['Camera Info'].pop('Port')

                cameras_info.append(dictionary)
        else :
            dictionary = dict()
            dictionary['Camera Name'] = CameraName
            dictionary['Camera Info'] = {}
            dictionary['Models'] = []
            cameras_info.append(dictionary)
    final_dictionary = {'cameras_data' : cameras_info}        
    #return  final_dictionary     
    return cameras_info
    
# pprint.pprint(all_cameras_info())

def counting_cameras_average_each_model():
    """
    Calculate the average count of all models for a list of cameras.

    Args:
        camera_names (list): List of camera names.

    Returns:
        pandas.DataFrame: DataFrame containing camera name, model name, and average count.
    """

    model_collection_mapping = {
        'crowdedDensity': 'ModelDensityData',
        'crossingBorder': 'ModelCountingData',
        'crowded': 'ModelCrowdedData',
        'vehicle': 'ModelVehicleData',
    }
    
    average_counts = []
    cameras_list  = finding_camera_names()
    for CameraName in cameras_list:
        for model_name, collection_name in model_collection_mapping.items():
            existing_collection = db[collection_name]

            query = {'Camera Info.Camera Name': CameraName}

            if check_existing_document(existing_collection, query):
                pipeline = [
                    {
                        "$match": query
                    },
                    {
                        "$group": {
                            "_id": "$Camera Info.Camera Name",
                            "count": {"$avg": "$Count"}
                        }
                    },
                    {
                        "$project": {
                            "Camera Name": "$_id",
                            "Count Average": "$count",
                            "Model": model_name,
                            "_id": 0
                        }
                    }
                ]

                try:
                    result = list(existing_collection.aggregate(pipeline))

                    if result:
                        result = result[0]
                        result['Count Average'] = math.ceil(result['Count Average'])
                        average_counts.append(result)
                except:
                    pass

            else :
                dictionary = {'Camera Name' : CameraName , 'Count Average': 0, 'Model': 'Not Found' }
                average_counts.append(dictionary)                                                    

    # Create DataFrame from the collected data
    df = pd.DataFrame(average_counts)
    final_dictionary = {'each_models_info' : average_counts}        
    #return     final_dictionary   
    return average_counts


# pprint.pprint(all_cameras_average_each_model())

def counting_cameras_average_frist_model():
    """
    Calculate the average count of all models for a list of cameras.

    Args:
        camera_names (list): List of camera names.

    Returns:
        pandas.DataFrame: DataFrame containing camera name, model name, and average count.
    """

    model_collection_mapping = {
        'crossingBorder': 'ModelCountingData',
        'crowdedDensity': 'ModelDensityData',
        'crowded': 'ModelCrowdedData',
    }
    
    average_counts = []
    cameras_list  = finding_camera_names()
    for CameraName in cameras_list:
        print('Getting Data of :' , CameraName)
        for model_name, collection_name in model_collection_mapping.items():
            existing_collection = db[collection_name]

            query = {'Camera Info.Camera Name': CameraName}

            if check_existing_document(existing_collection, query):
                pipeline = [
                    {
                        "$match": query
                    },
                    {
                        "$group": {
                            "_id": "$Camera Info.Camera Name",
                            "count": {"$avg": "$Count"}
                        }
                    },
                    {
                        "$project": {
                            "Camera Name": "$_id",
                            "Count Average": "$count",
                            "Model": model_name,
                            "_id": 0
                        }
                    }
                ]

                try:
                    result = list(existing_collection.aggregate(pipeline))

                    if result:
                        result = result[0]
                        result['Count Average'] = math.ceil(result['Count Average'])
                        average_counts.append(result)
                        break
                except:
                    pass

        else :
                dictionary = {'Camera Name' : CameraName , 'Count Average': 0, 'Model': 'Not Found' }
                average_counts.append(dictionary)                                                    

    # Create DataFrame from the collected data
    df = pd.DataFrame(average_counts)
    final_dictionary = {'each_models_info' : average_counts}        
    return    average_counts   

# pprint.pprint(all_cameras_average_frist_model())


def vehcile_counting_cameras_average():
    """
    Calculate the average count of all models for a list of cameras.

    Args:
        camera_names (list): List of camera names.

    Returns:
        pandas.DataFrame: DataFrame containing camera name, model name, and average count.
    """

    model_collection_mapping = {
        'vehicle': 'ModelVehicleData',

    }
    
    average_counts = []
    cameras_list  = finding_camera_names()
    for CameraName in cameras_list:
        print('Getting Data of :' , CameraName)
        for model_name, collection_name in model_collection_mapping.items():
            existing_collection = db[collection_name]

            query = {'Camera Info.Camera Name': CameraName}

            if check_existing_document(existing_collection, query):
                pipeline = [
                    {
                        "$match": query
                    },
                    {
                        "$group": {
                            "_id": "$Camera Info.Camera Name",
                            "count": {"$avg": "$Count"}
                        }
                    },
                    {
                        "$project": {
                            "Camera Name": "$_id",
                            "Count Average": "$count",
                            "Model": model_name,
                            "_id": 0
                        }
                    }
                ]

                try:
                    result = list(existing_collection.aggregate(pipeline))

                    if result:
                        result = result[0]
                        result['Count Average'] = math.ceil(result['Count Average'])
                        average_counts.append(result)
                        break
                except:
                    pass

        else :
                dictionary = {'Camera Name' : CameraName , 'Count Average': 0, 'Model': 'Not Found' }
                average_counts.append(dictionary)                                                    

    # Create DataFrame from the collected data
    df = pd.DataFrame(average_counts)
    final_dictionary = {'each_models_info' : average_counts}        
    return    average_counts   

# pprint.pprint(vehcile_counting_cameras_average())

def get_all_cameras_count_perH(day, month, year):
    """
    Filter data by date and get the average count in the form of time range for all models applied to a specific camera.

    Args:
        day (int): Day component of the date.
        month (int): Month component of the date.
        year (int): Year component of the date.

    Returns:
        list: List of dictionaries containing time range, model name, and average count for all cameras.
    """
    # Ensure month and day are zero-padded if less than 10
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)

    # Construct target date string
    target_date = f"{year}-{month_str}-{day_str}"

    # Map ModelName to corresponding collection
    model_collection_mapping = {
        'crossingBorder': 'ModelCountingData',
        'crowdedDensity': 'ModelDensityData',
        'crowded': 'ModelCrowdedData',
    }

    all_data = []
    camera_names = finding_camera_names()
    # camera_names = ['aslom', 'Ro7Elsharq' , 'io']
    
    try:

        for camera_name in camera_names:
            print('Getting Data of :' , camera_name)
            data = []  # Initialize data list for each camera
            for model_name, collection_name in model_collection_mapping.items():
                existing_collection = db[collection_name]

                query = {'Camera Info.Camera Name': camera_name, 'Date': target_date}

                if check_existing_document(existing_collection, query):
                    print(f'{camera_name} Camera Found in {model_name} Collection')
                    pipeline = [
                        {"$match": query},
                        {"$group": {"_id": {"$hour": "$Timestamp"}, "count": {"$avg": "$Count"}}},
                        {"$project": {"Hour": "$_id", "Count Average": {"$ceil": "$count"}, "_id": 0}},
                        {"$sort": {"Hour": 1}}
                    ]

                    result = list(existing_collection.aggregate(pipeline))
                    if result:
                        for item in result:
                            hour = item['Hour'] + 2
                            formatted_hour = hour if hour <= 12 else hour - 12
                            average_count = item['Count Average']
                            am_pm = "PM" if hour >= 12 else "AM"
                            time_range = f"{formatted_hour} {am_pm} - {formatted_hour + 1 if formatted_hour < 12 else 1} {am_pm}"
                            if time_range == '12 PM - 13 PM':
                                time_range = time_range.replace(time_range, '12 PM - 1 PM')
                            data.append({'Model': model_name, 'Time Range': time_range, 'Count Average': average_count})

                        for hour in range(24):
                            am_pm = "PM" if hour >= 12 else "AM"
                            formatted_hour = hour if hour <= 12 else hour - 12
                            time_range = f"{formatted_hour} {am_pm} - {formatted_hour + 1} {am_pm}"

                            data.append({'Model': model_name, 'Time Range': time_range, 'Count Average': 0})

                        break

            if not data:
                dictionary = {'Camera Name': camera_name, 'Time Range': target_date, 'Count Average': 0, 'Model': 'Not Found'}
            else:
                df = pd.DataFrame(data)
                df['Camera Name'] = camera_name

                # Define the order of time ranges
                df = df.sort_values(by=['Time Range', 'Count Average'], ascending=[True, False]).drop_duplicates(subset='Time Range')
                # Define the order of time ranges
                time_range_order = [f"{i} AM - {i + 1} AM" if i != 12 else f"{i} AM - {i + 1} PM" for i in range(12)]
                time_range_order.extend([f"{i} PM - {i + 1} PM" if i != 12 else f"{i} PM - {i + 1} AM" for i in range(12)])
                ind = time_range_order.index('0 PM - 1 PM')
                time_range_order[ind] = '12 PM - 1 PM'

                # Convert 'Time Range' column to categorical with predefined order
                df['Time Range'] = pd.Categorical(df['Time Range'], categories=time_range_order, ordered=True)

                # Sort DataFrame by 'Time Range'
                df.sort_values(by='Time Range', inplace=True)
                df.dropna(inplace=True)
                dictionary = df.to_dict(orient='records')
                all_data.append(dictionary)

        return all_data

    except Exception as e:
        print(e)
        dictionary = {'Camera Name': camera_name, 'Time Range': 'Null', 'Count Average': 0, 'Models': 'Not Found'}
        return dictionary

# pprint.pprint(get_all_cameras_count_perH('19','02','2024'))
    

    
#____________________________________________________________________________
def get_all_cameras_count_perM( month, year):
    """
    Filter data by date and get the average count per each day in a month for the specified camera.

    Args:
        month (string): Month component of the date.
        year (string): Year component of the date.

    Returns:
        list: List of dictionaries containing time range, model name, and average count for all cameras.
    """
    month_int = int(month)
    year = int(year)

    model_collection_mapping = {
        'crossingBorder': 'ModelCountingData',
        'crowded': 'ModelCrowdedData',
        'crowdedDensity': 'ModelDensityData',
    }

    # Ensure month and year are zero-padded if less than 10
    month_str = str(month_int).zfill(2)
    year_str = str(year)

    # Get the number of days in the specified month
    days_in_month = calendar.monthrange(year, month_int)[1]


    all_data = []
    camera_names = finding_camera_names()

    # camera_names = ['aslom', 'Ro7Elsharq' , 'io']

    try:
        for camera_name in camera_names:
            data = []
            print('Getting Data of :' , camera_name)

            data_found = False  # Flag to track whether data has been found
            camera_found = False  # Flag to track whether camera is found                
            for model_name, collection_name in model_collection_mapping.items():
                existing_collection = db[collection_name]
                if data_found:  # If data is already found, break the loop
                        break            
                for day in range(1, days_in_month + 1):
                    
                    day_str = str(day).zfill(2)
                    TargetDate = f"{year_str}-{month_str}-{day_str}"

                    query = {'Camera Info.Camera Name': camera_name, 'Date': TargetDate}

                    if check_existing_document(existing_collection, query):
                        camera_found = True  # Set the flag to True indicating camera is found
                        print(f'{camera_name} Camera Found in {model_name} Collection')
                        
                        pipeline = [
                            {"$match": query},
                            {"$group": {"_id": "$Date", "count": {"$avg": "$Count"}}},
                            {"$project": {"Date": "$_id", "Count Average": {"$ceil": "$count"}, "_id": 0}}
                        ]

                        result = list(existing_collection.aggregate(pipeline))

                        if result:
                            for item in result:
                                data.append({'Model': model_name, 'Time Range': item['Date'],
                                            'Camera Name': camera_name, 'Count Average': item['Count Average']})
                            data_found = True  # Set the flag to True indicating data is found
                        else:
                            # If no data available for this day, add a record with zero count
                            data.append({'Model': model_name, 'Time Range': TargetDate, 'Count Average': 0, 'Camera Name': camera_name})
                    else:
                            # If no data available for this day, add a record with zero count
                            data.append({'Model': model_name, 'Time Range': TargetDate, 'Count Average': 0, 'Camera Name': camera_name})                        

            if not camera_found:
                dictionary = {'Camera Name' :camera_name , 'Count Average' :0 , 'Model' : 'Not Found'  }
            
            else:
                df = pd.DataFrame(data)
                dictionary = df.to_dict(orient='records')
                all_data.append(dictionary)
        return all_data                 
    
    except Exception as e:
        print(e)
        return f"Error occurred: {str(e)}"
    
# pprint.pprint(get_all_cameras_count_perM(2,2024))    
    

#____________________________________________________________________________    
def get_all_cameras_count_perY(year):
    """
    Filter data by date and get the average count per each month in a year for the specified camera.

    Args:
        CameraName (str): Name of the camera.
        year (string): Year component of the date.

    Returns:
        pandas.DataFrame: DataFrame containing month, average count, camera name, and model for each month in the year.
    """
    year_int = int(year)

    model_collection_mapping = {
        'crowded': 'ModelCrowdedData',
        'crowdedDensity': 'ModelDensityData',
        'crossingBorder': 'ModelCountingData',
    }

    all_data = []
    camera_names = finding_camera_names()

    # camera_names = ['test','io','Ro7Elsharq']

    for CameraName in camera_names:
        data = []
        print('Getting Data of :' , CameraName)


        all_dfs = []  # List to store all DataFrames
        found = False
        # Iterate over the model collection mapping
        for modelname, collection_name in model_collection_mapping.items():
            if found :
                break
            existing_collection = db[collection_name] 

            # Ensure year is zero-padded if less than 10
            year_str = str(year_int)

            # Get the number of months in a year
            months_in_year = 12

            data = []

            for month in range(1, months_in_year + 1):
                month_str = str(month).zfill(2)

                query = {'Camera Info.Camera Name': CameraName, 'Date': {'$regex': f'^{year_str}-{month_str}-'}}

                if check_existing_document(existing_collection, query):
                    print(f'{CameraName} Camera Found in {modelname} Collection')

                    pipeline = [
                        {"$match": query},
                        {"$group": {"_id": {"$substr": ["$Date", 0, 10]}, "count": {"$avg": "$Count"}}},
                        {"$project": {"Date": "$_id", "Count Average": "$count", "_id": 0}}
                    ]

                    result = list(existing_collection.aggregate(pipeline))

                    if result:
                        for item in result:
                            data.append({'Month': calendar.month_name[int(month_str)], 'Count Average': item['Count Average']})
                            found = True                        
                    else:
                        # If no data available for this month, add a record with zero count
                        data.append({'Month':calendar.month_name[int(month_str)] , 'Count Average': 0})
                else : 
                        data.append({'Month':calendar.month_name[int(month_str)] , 'Count Average': 0})

            # Create DataFrame
            df = pd.DataFrame(data)
            try:
                average_per_month = df.groupby(df['Month']).mean()
                average_per_month.reset_index(inplace=True)  # Reset index
                average_per_month['Count Average'] = average_per_month['Count Average'].apply(math.ceil).astype(float)
                average_per_month['Camera Name'] = CameraName
                average_per_month['Model'] = modelname
                average_per_month.rename({'Month' : 'Time Range'} , axis=1,inplace=True)            
                all_dfs.append(average_per_month)  # Append DataFrame to the list
            except:
                df['Camera Name'] = CameraName
                df['Model'] = modelname
                all_dfs.append(df)  # Append DataFrame to the list

        # Concatenate all DataFrames
        final_df = pd.concat(all_dfs)
        if len(data) == 0 :
                dictionary = {'Camera Name' :CameraName , 'Count Average' :0 , 'Model' : 'Not Found'  }
        else :             
            # Sort by month order
            month_order = list(calendar.month_name)[1:]  # Remove the first element which is an empty string
            final_df['Month'] = pd.Categorical(final_df['Time Range'], categories=month_order, ordered=True)
            final_df.sort_values('Month', inplace=True)
            final_df.drop(columns='Month', inplace=True)  # Remove temporary 'Month' column used for sorting
            count_per_model = final_df.groupby('Model')['Count Average'].sum()
            model_lists = list(count_per_model.index)
            for model in model_lists : 
                if (count_per_model[model] == 0)   :
                # Drop rows where 'Model' column contains the specific value
                    final_df = final_df[final_df['Model'] != model]


        # Check if DataFrame is empty after dropping rows
            if len(final_df) == 0:
                final_df = pd.DataFrame({'Time Range': [year], 
                            'Count Average': [0], 
                            'Camera Name': [CameraName],
                            'Model': ['Not Found']})
                dictionary = final_df.to_dict(orient='records')
            else :
                dictionary = final_df.to_dict(orient='records')
                all_data.append(dictionary)
                
    return  all_data                     

# pprint.pprint(get_all_cameras_count_perY(2024))

def get_all_vechile_count_perH(day, month, year):
    """
    Filter data by date and get the average count in the form of time range for all models applied to a specific camera.

    Args:
        day (int): Day component of the date.
        month (int): Month component of the date.
        year (int): Year component of the date.

    Returns:
        List: List containing time range, model name, and average count.
    """
    # Ensure month and day are zero-padded if less than 10
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)

    # Construct target date string
    TargetDate = f"{year}-{month_str}-{day_str}"

    # Map ModelName to corresponding collection
    model_collection_mapping = {
        'vehicle': 'ModelVehicleData',
    }
    all_data = []
    # Camera_names = finding_camera_names()
    Camera_names =  all_cameras_in_model('vehicle')['Camera Names']

    # Camera_names =    ['aslom', 'VehicleTest' , 'io']    
    for CameraName in Camera_names :
        print('Getting Data of :' , CameraName)
        data = []
        try:
            for ModelName, collection_name in model_collection_mapping.items():
                existing_collection = db[collection_name]

                query = {'Camera Info.Camera Name': CameraName, 'Date': TargetDate}
                if check_existing_document(existing_collection, query):
                    print(f'{CameraName} Camera Found in {ModelName} Collection')
                    pipeline = [
                        {"$match": query},
                        {"$group": {"_id": {"$hour": "$Timestamp"}, "count": {"$avg": "$Count"}}},
                        {"$project": {"Hour": "$_id", "Count Average": {"$ceil": "$count"}, "_id": 0}},
                        {"$sort": {"Hour": 1}}
                    ]


                    result = list(existing_collection.aggregate(pipeline))
                    if result:
                        for item in result:
                            hour = item['Hour'] + 2
                            formatted_hour = hour if hour <= 12 else hour - 12
                            average_count = item['Count Average']
                            am_pm = "PM" if hour >= 12 else "AM"
                            time_range = f"{formatted_hour} {am_pm} - {formatted_hour + 1 if formatted_hour < 12 else 1} {am_pm}"
                            if time_range == '12 PM - 13 PM' :
                                    time_range = time_range.replace(time_range,'12 PM - 1 PM')                           
                            data.append({'Model': ModelName, 'Time Range': time_range, 'Count Average': average_count})


                        for hour in range(24):
                            am_pm = "PM" if (hour) >= 12 else "AM"
                            formatted_hour = (hour) if (hour) <= 12 else (hour) - 12
                            time_range = f"{formatted_hour} {am_pm} - {formatted_hour + 1} {am_pm}"


                            data.append({'Model': ModelName, 'Time Range': time_range, 'Count Average': 0})

                        break

            if len(data) == 0:
                dictionary = {'Camera Name': CameraName, 'Time Range': TargetDate, 'Count Average': 0, 'Model': 'Not Found'}
                # all_data.append(dictionary)
            else:
                df = pd.DataFrame(data)
                df['Camera Name'] = CameraName

                # Define the order of time ranges
                df = df.sort_values(by=['Time Range', 'Count Average'], ascending=[True, False]).drop_duplicates(subset='Time Range')
                # Define the order of time ranges
                time_range_order = [f"{i} AM - {i + 1} AM" if i != 12 else f"{i} AM - {i + 1} PM" for i in range(12)]
                time_range_order.extend([f"{i} PM - {i + 1} PM" if i != 12 else f"{i} PM - {i + 1} AM" for i in range(12)])
                ind =  time_range_order.index('0 PM - 1 PM')
                time_range_order[ind] = '12 PM - 1 PM'

                # Convert 'Time Range' column to categorical with predefined order
                df['Time Range'] = pd.Categorical(df['Time Range'], categories=time_range_order, ordered=True)

                # Sort DataFrame by 'Time Range'
                df.sort_values(by='Time Range', inplace=True)
                df.dropna(inplace=True)
                dictionary = df.to_dict(orient='records')
                all_data.append(dictionary)
                
        except Exception as e:
            print(e)
            dictionary = {'Camera Name': CameraName, 'Time Range': 'Null', 'Count Average': 0, 'Models': 'Not Found'}
            # all_data.append(dictionary)
    return all_data

# pprint.pprint(get_all_vechile_count_perH(28,2,2024))

def get_all_vechile_count_perM(month, year):

    """
    Filter data by date and get the average count per each day in a month for the specified camera.

    Args:
        month (string): Month component of the date.
        year (string): Year component of the date.

    Returns:
        List : List containing date and average count for each day in the month.
    """
    month_int = int(month)
    year = int(year)

    model_collection_mapping = {
        'vehicle': 'ModelVehicleData',
    }


    # Ensure month and year are zero-padded if less than 10
    month_str = str(month_int).zfill(2)
    year_str = str(year)

    # Get the number of days in the specified month
    days_in_month = calendar.monthrange(year, month_int)[1]


    all_data = []
    # Camera_names = finding_camera_names()
    Camera_names =  all_cameras_in_model('vehicle')['Camera Names']

    # Camera_names =    ['aslom', 'VehicleTest' , 'io']    
    for CameraName in Camera_names :
        print('Getting Data of :' , CameraName)
        data = []
        data_found = False  # Flag to track whether data has been found
        camera_found = False  # Flag to track whether camera is found

        try:
            for model_name, collection_name in model_collection_mapping.items():
                existing_collection = db[collection_name]
                if data_found:  # If data is already found, break the loop
                        break            
                for day in range(1, days_in_month + 1):
                    
                    day_str = str(day).zfill(2)
                    TargetDate = f"{year_str}-{month_str}-{day_str}"

                    query = {'Camera Info.Camera Name': CameraName, 'Date': TargetDate}

                    if check_existing_document(existing_collection, query):
                        camera_found = True  # Set the flag to True indicating camera is found
                        
                        pipeline = [
                            {"$match": query},
                            {"$group": {"_id": "$Date", "count": {"$avg": "$Count"}}},
                            {"$project": {"Date": "$_id", "Count Average": {"$ceil": "$count"}, "_id": 0}}
                        ]

                        result = list(existing_collection.aggregate(pipeline))

                        if result:
                            for item in result:
                                data.append({'Model': model_name, 'Time Range': item['Date'],
                                            'Camera Name': CameraName, 'Count Average': item['Count Average']})
                            data_found = True  # Set the flag to True indicating data is found
                        else:
                            # If no data available for this day, add a record with zero count
                            data.append({'Model': model_name, 'Time Range': TargetDate, 'Count Average': 0, 'Camera Name': CameraName})
                    else:
                            # If no data available for this day, add a record with zero count
                            data.append({'Model': model_name, 'Time Range': TargetDate, 'Count Average': 0, 'Camera Name': CameraName})                        

            if not camera_found:
                dictionary = {'Camera Name' :CameraName , 'Count Average' :0 , 'Model' : 'Not Found'  }
                # all_data.append(dictionary)            
            else:
                df= pd.DataFrame(data)
                dictionary = df.to_dict(orient='records')
                all_data.append(dictionary)            
        
        except Exception as e:
            print(e)
            return f"Error occurred: {str(e)}"
    return all_data

# pprint.pprint(get_all_vechile_count_perM(2,2024))

#____________________________________________________________________________    
def get_all_vechile_count_perY(year):
    """
    Filter data by date and get the average count per each month in a year for the specified camera.

    Args:
        year (string): Year component of the date.

    Returns:
        List: List containing month, average count, camera name, and model for each month in the year.
    """
    year_int = int(year)

    model_collection_mapping = {
        'vehicle': 'ModelVehicleData',
    }
    
    all_data = []
    # Camera_names = finding_camera_names()
    Camera_names =  all_cameras_in_model('vehicle')['Camera Names']
    # Camera_names =    ['aslom', 'VehicleTest' , 'io']    
    for CameraName in Camera_names :
        print('Getting Data of :' , CameraName)
        all_dfs = []  # List to store all DataFrames
        found = False
        # Iterate over the model collection mapping
        for modelname, collection_name in model_collection_mapping.items():
            if found :
                break
            existing_collection = db[collection_name] 

            # Ensure year is zero-padded if less than 10
            year_str = str(year_int)

            # Get the number of months in a year
            months_in_year = 12

            data = []

            for month in range(1, months_in_year + 1):
                month_str = str(month).zfill(2)

                query = {'Camera Info.Camera Name': CameraName, 'Date': {'$regex': f'^{year_str}-{month_str}-'}}

                if check_existing_document(existing_collection, query):
                    pipeline = [
                        {"$match": query},
                        {"$group": {"_id": {"$substr": ["$Date", 0, 10]}, "count": {"$avg": "$Count"}}},
                        {"$project": {"Date": "$_id", "Count Average": "$count", "_id": 0}}
                    ]

                    result = list(existing_collection.aggregate(pipeline))

                    if result:
                        for item in result:
                            data.append({'Month': calendar.month_name[int(month_str)], 'Count Average': item['Count Average']})
                            found = True                        
                    else:
                        # If no data available for this month, add a record with zero count
                        data.append({'Month':calendar.month_name[int(month_str)] , 'Count Average': 0})
                else : 
                        data.append({'Month':calendar.month_name[int(month_str)] , 'Count Average': 0})

            # Create DataFrame
            df = pd.DataFrame(data)
            try:
                average_per_month = df.groupby(df['Month']).mean()
                average_per_month.reset_index(inplace=True)  # Reset index
                average_per_month['Count Average'] = average_per_month['Count Average'].apply(math.ceil).astype(float)
                average_per_month['Camera Name'] = CameraName
                average_per_month['Model'] = modelname
                average_per_month.rename({'Month' : 'Time Range'} , axis=1,inplace=True)            
                all_dfs.append(average_per_month)  # Append DataFrame to the list
            except:
                df['Camera Name'] = CameraName
                df['Model'] = modelname
                all_dfs.append(df)  # Append DataFrame to the list

        # Concatenate all DataFrames
        final_df = pd.concat(all_dfs)
        if len(data) == 0 :
                dictionary = {'Camera Name' :CameraName , 'Count Average' :0 , 'Model' : 'Not Found'  }
                # all_data.append(dictionary)
        else :             
            # Sort by month order
            month_order = list(calendar.month_name)[1:]  # Remove the first element which is an empty string
            final_df['Month'] = pd.Categorical(final_df['Time Range'], categories=month_order, ordered=True)
            final_df.sort_values('Month', inplace=True)
            final_df.drop(columns='Month', inplace=True)  # Remove temporary 'Month' column used for sorting
            count_per_model = final_df.groupby('Model')['Count Average'].sum()
            model_lists = list(count_per_model.index)
            for model in model_lists : 
                if (count_per_model[model] == 0)   :
                # Drop rows where 'Model' column contains the specific value
                    final_df = final_df[final_df['Model'] != model]


        # Check if DataFrame is empty after dropping rows
            if len(final_df) == 0:
                final_df = pd.DataFrame({'Time Range': [year], 
                            'Count Average': [0], 
                            'Camera Name': [CameraName],
                            'Model': ['Not Found']})
                dictionary = final_df.to_dict(orient='records')
                # all_data.append(dictionary)
            else :
                dictionary = final_df.to_dict(orient='records')
                all_data.append(dictionary)
                        

    return all_data                
         
# pprint.pprint(get_all_vechile_count_perY(2024))

def save_camera_info(camera_info):
    # Create the collection dynamically
    collection_name = "camera_info"
    collection = db[collection_name]
    collection.insert_one(camera_info)
    return collection_name

# def watch_changes():
#     with db.watch(full_document='updateLookup') as change_stream:
#         for change in change_stream:
#             if change['operationType'] == 'insert':
#                 print("New Document Inserted:", change['fullDocument'])
#                 # Check if 'Camera Info' exists before accessing its subfields
#                 camera_info = {
#                     "Camera Name": change['fullDocument'].get('Camera Info', {}).get('Camera Name'),
#                     "Model Name": change['fullDocument'].get('Model Name')
#                 }
#                 # Save the camera info only when explicitly called
#                 return save_camera_info(camera_info)

# collection_name = watch_changes()
# print(f"Collection '{collection_name}' created and insertion saved.")

def gender_filtering_date_aggrigates_only_Gender(CameraName, day, month, year):
    
    """
    Filter data by date and get the average count in the form of time range.

    Args:
        CameraName (str): Name of the camera.
        day (int): Day component of the date.
        month (int): Month component of the date.
        year (int): Year component of the date.

    Returns:
        pandas.DataFrame: DataFrame containing time range, total count for male, and total count for female.
    """
    # Ensure month and day are zero-padded if less than 10
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)

    # Construct target date string
    TargetDate = f"{year}-{month_str}-{day_str}"

    existing_collection = db['ModelGenderData']
    
    # Query to filter by camera name and date
    query = {'Camera Info.Camera Name': CameraName, 'Date': TargetDate}
    data = []
    if check_existing_document(existing_collection, query):
        print(f'{CameraName} Camera Found in Gender Collection')
        
        pipeline = [
            {"$match": query},
            {"$unwind": "$Label"},
            {"$group": {
                "_id": {
                    "Hour": {"$hour": "$Timestamp"},
                    "Gender": "$Label.Gender"
                },
                "count": {"$sum": 1}
            }},
            {"$project": {
                "Hour": "$_id.Hour",
                "Gender": "$_id.Gender",
                "Count": "$count",
                "_id": 0
            }},
            {"$sort": {"Hour": 1}}
        ]


        result = list(existing_collection.aggregate(pipeline))
        if result:
            
            for item in result:
                hour = item['Hour']
                gender = item['Gender']
                count = item['Count']
                
                # Adjusting hour for display
                am_pm = "PM" if (hour + 2) >= 12 else "AM"
                formatted_hour = (hour + 2) if (hour + 2) <= 12 else (hour + 2) - 12
                time_range = f"{formatted_hour} {am_pm} - {formatted_hour + 1} {am_pm}"
                
                data.append({
                    'Time Range': time_range,
                    'Gender': gender,
                    'Count': count,
                })

                for hour in range(24):
                        am_pm = "PM" if (hour) >= 12 else "AM"
                        formatted_hour = (hour) if (hour) <= 12 else (hour) - 12
                        time_range = f"{formatted_hour} {am_pm} - {formatted_hour + 1} {am_pm}"
                
                        data.append({
                            'Time Range': time_range,
                            'Gender': gender,
                            'Count': 0,
                        })

                        
            
            # Create DataFrame from collected data
            df = pd.DataFrame(data)
            
            # Pivot DataFrame to get male and female counts separately
            df_pivot = df.pivot_table(index='Time Range', columns='Gender', values='Count', aggfunc='sum', fill_value=0)
            
            # Reset index to make 'Time Range' a column
            df_pivot.reset_index(inplace=True)
            
            # Rename columns for clarity
            df_pivot.columns.name = None
            
            try :                                        
                    try :
                    # Add a total count column
                        df_pivot['Total Count'] = df_pivot['Female'] + df_pivot['Male']
                        df_pivot['Camera Name'] = CameraName   
                    except : 
                        df_pivot['Total Count'] = df_pivot['Female'] 
                        df_pivot['Female'] = 0                        
                        df_pivot['Camera Name'] = CameraName                   
            except :
                        df_pivot['Total Count'] = df_pivot['Male'] 
                        df_pivot['Female'] = 0                        

                        df_pivot['Camera Name'] = CameraName              
            

    if len(data) == 0:
        pass
    else :

            # Define the order of time ranges
            df = df_pivot.sort_values(by=['Time Range', 'Total Count'], ascending=[True, False]).drop_duplicates(subset='Time Range')
            # Define the order of time ranges
            time_range_order = [f"{i} AM - {i + 1} AM" if i != 12 else f"{i} AM - {i + 1} PM" for i in range(12)]
            time_range_order.extend([f"{i} PM - {i + 1} PM" if i != 12 else f"{i} PM - {i + 1} AM" for i in range(12)])
            ind =  time_range_order.index('0 PM - 1 PM')
            time_range_order[ind] = '12 PM - 1 PM'

            # Convert 'Time Range' column to categorical with predefined order
            df['Time Range'] = pd.Categorical(df['Time Range'], categories=time_range_order, ordered=True)

            # Sort DataFrame by 'Time Range'
            df.sort_values(by='Time Range', inplace=True)
            df.dropna(inplace=True)
            return df 
# gender_filtering_date_aggrigates_only_Gender('aslom',9,3,2024)

def get_all_cameras_genderPerH( day, month, year):
    
    """
    Filter data by date and get the average count in the form of time range.

    Args:
        day (int): Day component of the date.
        month (int): Month component of the date.
        year (int): Year component of the date.

    Returns:
        List: List containing month, total count, camera name, and model for each month in the year.
    """
    # Ensure month and day are zero-padded if less than 10
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)

    # Construct target date string
    TargetDate = f"{year}-{month_str}-{day_str}"

    existing_collection = db['ModelGenderData']
    # camera_names = finding_camera_names()
    camera_names =  all_cameras_in_model('gender')['Camera Names']
    # camera_names = ['aslom','GenderTest','FristCam']
    all_data = []
    for CameraName in camera_names :
        print('Getting Data of :' , CameraName)
        # Query to filter by camera name and date
        query = {'Camera Info.Camera Name': CameraName, 'Date': TargetDate}
        data = []
        if check_existing_document(existing_collection, query):
            print(f'{CameraName} Camera Found in Gender Collection')
            
            pipeline = [
                {"$match": query},
                {"$unwind": "$Label"},
                {"$group": {
                    "_id": {
                        "Hour": {"$hour": "$Timestamp"},
                        "Gender": "$Label.Gender"
                    },
                    "count": {"$sum": 1}
                }},
                {"$project": {
                    "Hour": "$_id.Hour",
                    "Gender": "$_id.Gender",
                    "Count": "$count",
                    "_id": 0
                }},
                {"$sort": {"Hour": 1}}
            ]


            result = list(existing_collection.aggregate(pipeline))
            if result:
                
                for item in result:
                    hour = item['Hour']
                    gender = item['Gender']
                    count = item['Count']
                    
                    # Adjusting hour for display
                    am_pm = "PM" if (hour + 2) >= 12 else "AM"
                    formatted_hour = (hour + 2) if (hour + 2) <= 12 else (hour + 2) - 12
                    time_range = f"{formatted_hour} {am_pm} - {formatted_hour + 1} {am_pm}"
                    
                    data.append({
                        'Time Range': time_range,
                        'Gender': gender,
                        'Count': count,
                    })

                    for hour in range(24):
                            am_pm = "PM" if (hour) >= 12 else "AM"
                            formatted_hour = (hour) if (hour) <= 12 else (hour) - 12
                            time_range = f"{formatted_hour} {am_pm} - {formatted_hour + 1} {am_pm}"
                    
                            data.append({
                                'Time Range': time_range,
                                'Gender': gender,
                                'Count': 0,
                            })

                            
                
                # Create DataFrame from collected data
                df = pd.DataFrame(data)
                # print(df)    
                # Pivot DataFrame to get male and female counts separately
                df_pivot = df.pivot_table(index='Time Range', columns='Gender', values='Count', aggfunc='sum', fill_value=0)
                
                # Reset index to make 'Time Range' a column
                df_pivot.reset_index(inplace=True)
                
                # Rename columns for clarity
                df_pivot.columns.name = None
                try :                                        
                    try :
                    # Add a total count column
                        df_pivot['Total Count'] = df_pivot['Female'] + df_pivot['Male']
                        df_pivot['Camera Name'] = CameraName   
                    except : 
                        df_pivot['Male'] = 0
                        df_pivot['Total Count'] = df_pivot['Female'] 
                        df_pivot['Camera Name'] = CameraName                   
                except :
                        df_pivot['Female'] = 0                        
                        df_pivot['Total Count'] = df_pivot['Male'] 
                        df_pivot['Camera Name'] = CameraName               

        if len(data) == 0:
                    dictionary = {'Camera Name': CameraName, 'Time Range': TargetDate,
                                'Female' : 0 ,'Male' :0, 'Total Count': 0}
                    # all_data.append(dictionary)
        else :

                # Define the order of time ranges
                df = df_pivot.sort_values(by=['Time Range', 'Total Count'], ascending=[True, False]).drop_duplicates(subset='Time Range')
                # Define the order of time ranges
                time_range_order = [f"{i} AM - {i + 1} AM" if i != 12 else f"{i} AM - {i + 1} PM" for i in range(12)]
                time_range_order.extend([f"{i} PM - {i + 1} PM" if i != 12 else f"{i} PM - {i + 1} AM" for i in range(12)])
                ind =  time_range_order.index('0 PM - 1 PM')
                time_range_order[ind] = '12 PM - 1 PM'

                # Convert 'Time Range' column to categorical with predefined order
                df['Time Range'] = pd.Categorical(df['Time Range'], categories=time_range_order, ordered=True)

                # Sort DataFrame by 'Time Range'
                df.sort_values(by='Time Range', inplace=True)
                df.dropna(inplace=True)
                dictionary = df.to_dict(orient='records')       
                all_data.append(dictionary)

    return all_data
# pprint.pprint(get_all_cameras_genderPerH(9,3,2024))   
    
def get_all_cameras_genderPerM(month, year):

    """
    Filter data by month and get the total count for male and female for each day.

    Args:
        month (int): Month component of the date.
        year (int): Year component of the date.

    Returns:
        List: List containing month, total count, camera name, and model for each month in the year.
    """
    # Ensure month is zero-padded if less than 10
    month = int(month)
    year = int(year)
    month_str = str(month).zfill(2)
    # Get the number of days in the month
    num_days = calendar.monthrange(year, month)[1]
    # camera_names = finding_camera_names()
    camera_names =  all_cameras_in_model('gender')['Camera Names']
    all_data = []

    if isinstance(camera_names, list):  
    # camera_names = ['aslom','GenderTest','FristCam']
        for CameraName in camera_names :
            print('Getting Data of :' , CameraName)
            # Query to filter by camera name and date    
            # Initialize an empty list to store data for all days in the month
            data = []
            # Iterate over all days in the month
            for day in range(1, num_days + 1):
                # Retrieve data for the specific day using the existing function
                daily_data = gender_filtering_date_aggrigates(CameraName, day, month, year)
                # print(daily_data)                
                # If there's no data for the day, add zeros for male and female counts
                if isinstance(daily_data, dict): # If there's no data for the day
                    data.append({
                        'Camera Name': CameraName,
                        'Time Range': f"{year}-{month_str}-{str(day).zfill(2)}",
                        'Male': 0,
                        'Female': 0,
                        'Total Count': 0
                    })
                else:
                    # Calculate the total count for male and female for the day
                    male_count = daily_data['Male'].sum()
                    female_count = daily_data['Female'].sum()
                    total_count = male_count + female_count
                    
                    # Add the data for the day to the list
                    data.append({
                        'Camera Name': CameraName,
                        'Time Range': f"{year}-{month_str}-{str(day).zfill(2)}",
                        'Male': male_count,
                        'Female': female_count,
                        'Total Count': total_count
                    })

            # Convert the list of dictionaries to a DataFrame
            df = pd.DataFrame(data)

            # Return the DataFrame
            dictionary = df.to_dict(orient='records')
            all_data.append(dictionary)

    return all_data            

# pprint.pprint(get_all_cameras_genderPerM(3,2024))        

def get_all_cameras_genderPerY(year):
    """
    Filter data by year and get the total count for male and female for each month.

    Args:
        CameraName (str): Name of the camera.
        year (int): Year component.

    Returns:
        pandas.DataFrame: DataFrame containing total count for male and female for each month.
    """
    # Initialize an empty list to store data for all months in the year
    year = int(year)
    camera_names =  all_cameras_in_model('gender')['Camera Names']
    all_data = []
    for CameraName in camera_names :
        data = []
        # Iterate over all months in the year
        for month in range(1, 13):
            # Get the number of days in the month
            num_days = calendar.monthrange(year, month)[1]

            # Initialize counters for male and female counts for the month
            male_count_month = 0
            female_count_month = 0

            # Iterate over all days in the month
            for day in range(1, num_days + 1):
                # Retrieve data for the specific day using the existing function
                daily_data = gender_filtering_date_aggrigates(CameraName, day, month, year)
                
                # If there's data for the day, sum the male and female counts
                if not isinstance(daily_data, dict): # If there's no data for the day
                    male_count_month += daily_data['Male'].sum()
                    female_count_month += daily_data['Female'].sum()

            # Calculate the total count for the month
            total_count_month = male_count_month + female_count_month
            
            # Add the aggregated data for the month to the list
            data.append({
                'Camera Name': CameraName,
                'Time Range': calendar.month_name[int(month)],
                'Year' : year ,
                'Male': male_count_month,
                'Female': female_count_month,
                'Total Count': total_count_month
            })

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(data)
        dictonary =    df.to_dict(orient='records')
        all_data.append(dictonary)
        # Return the DataFrame
    return all_data


# pprint.pprint(get_all_cameras_genderPerY(2024))    


def all_cameras_in_violence():
    # Specify the collection names directly
    collection_name = 'ModelViolenceData'
    

    
    # Check if the collection exists
    if collection_name not in db.list_collection_names():
        camera_names = distinct_camera_names
        return camera_names

    existing_collection = db[collection_name]
    
    distinct_camera_names = existing_collection.distinct('Camera Info.Camera Name')

    camera_names =distinct_camera_names
    return camera_names

def all_running_now_data():
    existing_collection = db['RunningNow']
    result = existing_collection.find()
    
    model_names_dict = {}
    for doc in result:
        camera_name = doc.get('Camera Name')
        model_name = doc.get('Model Name')
        
        if camera_name not in model_names_dict:
            model_names_dict[camera_name] = [model_name]
        else:
            model_names_dict[camera_name].append(model_name)
    
    running_now_data = [{'Camera Name': camera, 'Models Applied': model_list} for camera, model_list in model_names_dict.items()]
    return running_now_data

# running_now_data = all_running_now_data()
# print(running_now_data)

# for data in running_now_data :
#     print(data)
#     if (data['Camera Name'] == 'aslom') :
#         list__ = data['Models Applied']
#         items_to_remove = ['gender']
#         updated_list = [item for item in list__ if item not in items_to_remove]
#         print(updated_list)


def update_camera_status_models_collection(CameraName):
    
    collection_mapping = {
        'violence': 'ModelViolenceData',
        'vehicle': 'ModelVehicleData',
        'crowdedDensity': 'ModelDensityData',
        'crossingBorder': 'ModelCountingData',
        'crowded': 'ModelCrowdedData',
        'gender': 'ModelGenderData',
        'clothes color': 'ModelClothesColorData'
    }
    new_status = 'OFF'
    for model_name, collection_name in collection_mapping.items():
        existing_collection = db[collection_name] 
        query = {'Camera Info.Camera Name': CameraName}
        if check_existing_document(existing_collection, query):
            update_query = {"$set": {"Camera Info.Status": new_status}}
            existing_collection.update_many(query, update_query)
            print(f"Updated status for colllection '{model_name}' documents containing camera '{CameraName}' to '{new_status}'.")            



# update_camera_status_models_collection('VoilenceTest')

def update_camera_status_specific_models(CameraName, models):
    collection_mapping = {
        'violence': 'ModelViolenceData',
        'vehicle': 'ModelVehicleData',
        'crowdedDensity': 'ModelDensityData',
        'crossingBorder': 'ModelCountingData',
        'crowded': 'ModelCrowdedData',
        'gender': 'ModelGenderData',
        'clothes color': 'ModelClothesColorData'
    }
    new_status = 'OFF'
    for model_name in models:
        if model_name in collection_mapping:
            collection_name = collection_mapping[model_name]
            existing_collection = db[collection_name]
            query = {'Camera Info.Camera Name': CameraName}
            if check_existing_document(existing_collection, query):
                update_query = {"$set": {"Camera Info.Status": new_status}}
                existing_collection.update_many(query, update_query)
                print(f"Updated status for collection '{model_name}' documents containing camera '{CameraName}' to '{new_status}'.")
        else:
            print(f"Model '{model_name}' not found in the collection mapping.")


def postcam_getallmodelsperMinutes(CameraName, hour, day, month, year):
    """
    Filter data by date and hour and get the maximum count in the form of time range for all models applied to a specific camera, grouped by minutes within the specified hour.

    Args:
        CameraName (str): Name of the camera.
        hour (int): Hour component of the time range.
        day (int): Day component of the date.
        month (int): Month component of the date.
        year (int): Year component of the date.

    Returns:
        list of dicts: List containing dictionaries with 'Camera Name', 'Time Range', 'Model', and 'Max Count' keys.
    """
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)

    TargetDate = f"{year}-{month_str}-{day_str}"

    model_collection_mapping = {
        'crossingBorder': 'ModelCountingData',
        'crowdedDensity': 'ModelDensityData',
        'crowded': 'ModelCrowdedData',
    }

    hour = int(hour)
    hour = hour - 2
    data = []
    try:
        for ModelName, collection_name in model_collection_mapping.items():
            existing_collection = db[collection_name]

            query = {
                'Camera Info.Camera Name': CameraName,
                'Date': TargetDate,
                '$expr': {'$eq': [{'$hour': '$Timestamp'}, hour]}
            }

            if check_existing_document(existing_collection, query):
                print(f'{CameraName} Camera Found in {ModelName} Collection')
                pipeline = [
                    {"$match": query},
                    {"$group": {
                        "_id": {
                            "minute": {"$minute": "$Timestamp"}
                        },
                        "maxCount": {"$max": "$Count"}
                    }},
                    {"$project": {
                        "Minute": "$_id.minute",
                        "Max Count": "$maxCount",
                        "_id": 0
                    }}
                ]

                result = list(existing_collection.aggregate(pipeline))
                # print(result)
                if result:
                    for item in result:
                        minute = item['Minute']
                        max_count = item['Max Count']
                        time_range = f"{hour + 2:02}:{minute:02}"
                        data.append({'Camera Name': CameraName, 'Time Range': time_range, 'Model': ModelName, 'Count Average': max_count})

                    for minute in range(60):
                        time_range = f"{hour + 2:02}:{minute:02}"
                        data.append({'Camera Name': CameraName, 'Time Range': time_range, 'Model': ModelName, 'Count Average': 0})

                    break

        if len(data) == 0:
            dictionary = {'Camera Name': CameraName, 'Time Range': TargetDate, 'Count Average': 0, 'Model': 'Not Found'}
            return dictionary
        else:
            data.sort(key=lambda x: x['Time Range'])
            df = pd.DataFrame(data)
            df = df.sort_values(by=['Time Range', 'Count Average'], ascending=[True, False]).drop_duplicates(subset='Time Range')
            dictionary = df.to_dict(orient='records')
            return dictionary

    except Exception as e:
        print(e)
        dictionary = {'Camera Name': CameraName, 'Time Range': 'Null', 'Count Average': 0, 'Model': 'Not Found'}
        return dictionary



# pprint.pprint(postcam_getallmodelsperMinutes('MinutesTesting',11,25,3,2024))


def get_all_cameras_count_perMinutes(hour, day, month, year):
    """
    Filter data by date and hour and get the maximum count in the form of time range for all models applied to a specific camera, grouped by minutes within the specified hour.

    Args:
        CameraName (str): Name of the camera.
        hour (int): Hour component of the time range.
        day (int): Day component of the date.
        month (int): Month component of the date.
        year (int): Year component of the date.

    Returns:
        list of dicts: List containing dictionaries with 'Camera Name', 'Time Range', 'Model', and 'Max Count' keys.
    """
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)

    TargetDate = f"{year}-{month_str}-{day_str}"

    model_collection_mapping = {
        'crossingBorder': 'ModelCountingData',
        'crowdedDensity': 'ModelDensityData',
        'crowded': 'ModelCrowdedData',
    }

    hour = int(hour)
    hour = hour - 2
    all_data = []
    camera_names = finding_camera_names()
    # camera_names = ['aslom', 'MinutesTesting']    
    try:
        for CameraName in camera_names:
            print('Getting Data of :' , CameraName)
            data = []  # Initialize data list for each camera        
            for ModelName, collection_name in model_collection_mapping.items():
                existing_collection = db[collection_name]

                query = {
                    'Camera Info.Camera Name': CameraName,
                    'Date': TargetDate,
                    '$expr': {'$eq': [{'$hour': '$Timestamp'}, hour]}
                }

                if check_existing_document(existing_collection, query):
                    print(f'{CameraName} Camera Found in {ModelName} Collection')
                    pipeline = [
                        {"$match": query},
                        {"$group": {
                            "_id": {
                                "minute": {"$minute": "$Timestamp"}
                            },
                            "maxCount": {"$max": "$Count"}
                        }},
                        {"$project": {
                            "Minute": "$_id.minute",
                            "Max Count": "$maxCount",
                            "_id": 0
                        }}
                    ]

                    result = list(existing_collection.aggregate(pipeline))
                    print(result)
                    if result:
                        for item in result:
                            minute = item['Minute']
                            max_count = item['Max Count']
                            time_range = f"{hour + 2:02}:{minute:02}"
                            data.append({'Camera Name': CameraName, 'Time Range': time_range, 'Model': ModelName, 'Count Average': max_count})

                        for minute in range(60):
                            time_range = f"{hour + 2:02}:{minute:02}"
                            data.append({'Camera Name': CameraName, 'Time Range': time_range, 'Model': ModelName, 'Count Average': 0})

                        break

            if len(data) == 0:
                dictionary = {'Camera Name': CameraName, 'Time Range': TargetDate, 'Count Average': 0, 'Model': 'Not Found'}
                # return dictionary
            else:
                data.sort(key=lambda x: x['Time Range'])
                df = pd.DataFrame(data)
                df = df.sort_values(by=['Time Range', 'Count Average'], ascending=[True, False]).drop_duplicates(subset='Time Range')
                dictionary = df.to_dict(orient='records')                
                all_data.append(dictionary)
        return   all_data              

    except Exception as e:
        print(e)
        dictionary = {'Camera Name': CameraName, 'Time Range': 'Null', 'Count Average': 0, 'Model': 'Not Found'}
        return dictionary

# pprint.pprint(get_all_cameras_count_perMinutes(11,25,3,2024))
    
def postcam_getvechileMinutes(CameraName, hour, day, month, year):
    """
    Filter data by date and hour and get the maximum count in the form of time range for all models applied to a specific camera, grouped by minutes within the specified hour.

    Args:
        CameraName (str): Name of the camera.
        hour (int): Hour component of the time range.
        day (int): Day component of the date.
        month (int): Month component of the date.
        year (int): Year component of the date.

    Returns:
        list of dicts: List containing dictionaries with 'Camera Name', 'Time Range', 'Model', and 'Max Count' keys.
    """
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)

    TargetDate = f"{year}-{month_str}-{day_str}"

    model_collection_mapping = {
        'vehicle': 'ModelVehicleData',
    }   
    hour = int(hour)
    hour = hour - 2
    data = []
    try:
        for ModelName, collection_name in model_collection_mapping.items():
            existing_collection = db[collection_name]

            query = {
                'Camera Info.Camera Name': CameraName,
                'Date': TargetDate,
                '$expr': {'$eq': [{'$hour': '$Timestamp'}, hour]}
            }

            if check_existing_document(existing_collection, query):
                print(f'{CameraName} Camera Found in {ModelName} Collection')
                pipeline = [
                    {"$match": query},
                    {"$group": {
                        "_id": {
                            "minute": {"$minute": "$Timestamp"}
                        },
                        "maxCount": {"$max": "$Count"}
                    }},
                    {"$project": {
                        "Minute": "$_id.minute",
                        "Max Count": "$maxCount",
                        "_id": 0
                    }}
                ]

                result = list(existing_collection.aggregate(pipeline))
                print(result)
                if result:
                    for item in result:
                        minute = item['Minute']
                        max_count = item['Max Count']
                        time_range = f"{hour + 2:02}:{minute:02}"
                        data.append({'Camera Name': CameraName, 'Time Range': time_range, 'Model': ModelName, 'Count Average': max_count})

                    for minute in range(60):
                        time_range = f"{hour + 2:02}:{minute:02}"
                        data.append({'Camera Name': CameraName, 'Time Range': time_range, 'Model': ModelName, 'Count Average': 0})

                    break

        if len(data) == 0:
            dictionary = {'Camera Name': CameraName, 'Time Range': TargetDate, 'Count Average': 0, 'Model': 'Not Found'}
            return dictionary
        else:
            data.sort(key=lambda x: x['Time Range'])
            df = pd.DataFrame(data)
            df = df.sort_values(by=['Time Range', 'Count Average'], ascending=[True, False]).drop_duplicates(subset='Time Range')
            dictionary = df.to_dict(orient='records')
            return dictionary

    except Exception as e:
        print(e)
        dictionary = {'Camera Name': CameraName, 'Time Range': 'Null', 'Count Average': 0, 'Model': 'Not Found'}
        return dictionary    
    

# pprint.pprint(postcam_getvechileMinutes('MinutesTesting',12,25,3,2024))
    
def get_all_vechile_count_perMinutes(hour, day, month, year):
    """
    Filter data by date and hour and get the maximum count in the form of time range for all models applied to a specific camera, grouped by minutes within the specified hour.

    Args:
        CameraName (str): Name of the camera.
        hour (int): Hour component of the time range.
        day (int): Day component of the date.
        month (int): Month component of the date.
        year (int): Year component of the date.

    Returns:
        list of dicts: List containing dictionaries with 'Camera Name', 'Time Range', 'Model', and 'Max Count' keys.
    """
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)

    TargetDate = f"{year}-{month_str}-{day_str}"
    model_collection_mapping = {
        'vehicle': 'ModelVehicleData',
    }   
    hour = int(hour)
    hour = hour - 2
    all_data = []
    camera_names = finding_camera_names()
    # camera_names = ['aslom', 'MinutesTesting']    
    try:
        for CameraName in camera_names:
            print('Getting Data of :' , CameraName)
            data = []  # Initialize data list for each camera        
            for ModelName, collection_name in model_collection_mapping.items():
                existing_collection = db[collection_name]

                query = {
                    'Camera Info.Camera Name': CameraName,
                    'Date': TargetDate,
                    '$expr': {'$eq': [{'$hour': '$Timestamp'}, hour]}
                }

                if check_existing_document(existing_collection, query):
                    print(f'{CameraName} Camera Found in {ModelName} Collection')
                    pipeline = [
                        {"$match": query},
                        {"$group": {
                            "_id": {
                                "minute": {"$minute": "$Timestamp"}
                            },
                            "maxCount": {"$max": "$Count"}
                        }},
                        {"$project": {
                            "Minute": "$_id.minute",
                            "Max Count": "$maxCount",
                            "_id": 0
                        }}
                    ]

                    result = list(existing_collection.aggregate(pipeline))
                    print(result)
                    if result:
                        for item in result:
                            minute = item['Minute']
                            max_count = item['Max Count']
                            time_range = f"{hour + 2:02}:{minute:02}"
                            data.append({'Camera Name': CameraName, 'Time Range': time_range, 'Model': ModelName, 'Count Average': max_count})

                        for minute in range(60):
                            time_range = f"{hour + 2:02}:{minute:02}"
                            data.append({'Camera Name': CameraName, 'Time Range': time_range, 'Model': ModelName, 'Count Average': 0})

                        break

            if len(data) == 0:
                dictionary = {'Camera Name': CameraName, 'Time Range': TargetDate, 'Count Average': 0, 'Model': 'Not Found'}
                # return dictionary
            else:
                data.sort(key=lambda x: x['Time Range'])
                df = pd.DataFrame(data)
                df = df.sort_values(by=['Time Range', 'Count Average'], ascending=[True, False]).drop_duplicates(subset='Time Range')
                dictionary = df.to_dict(orient='records')                
                all_data.append(dictionary)
        return  all_data              

    except Exception as e:
        print(e)
        dictionary = {'Camera Name': CameraName, 'Time Range': 'Null', 'Count Average': 0, 'Model': 'Not Found'}
        return dictionary
    
# pprint.pprint(get_all_vechile_count_perMinutes(12,25,3,2024))

def gender_filtering_Minutes_aggregates(CameraName, hour, day, month, year):
    """
    Filter data by date and hour, and get the count for male, female, and total count for each minute within that hour.

    Args:
        CameraName (str): Name of the camera.
        hour (int): Hour for which data is to be aggregated.
        day (int): Day component of the date.
        month (int): Month component of the date.
        year (int): Year component of the date.

    Returns:
        pandas.DataFrame: DataFrame containing minute-wise count for male, female, and total count for the given hour.
    """
    # Ensure month, day, and hour are zero-padded if less than 10
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)
    hour = int(hour) 
    hour -=2
    # Construct target date string
    TargetDate = f"{year}-{month_str}-{day_str}"

    existing_collection = db['ModelGenderData']

    # Query to filter by camera name, date, and hour
    query = {
        'Camera Info.Camera Name': CameraName,
        'Date': TargetDate,
        '$expr': {'$eq': [{'$hour': '$Timestamp'}, hour]}
    }

    data = []
    if check_existing_document(existing_collection, query):
        print(f'{CameraName} Camera Found in Gender Collection')

        pipeline = [
            {"$match": query},
            {"$unwind": "$Label"},
            {"$group": {
                "_id": {
                    "Minute": {"$minute": "$Timestamp"},
                    "Gender": "$Label.Gender"
                },
                "count": {"$sum": 1}
            }},
            {"$project": {
                "Minute": "$_id.Minute",
                "Gender": "$_id.Gender",
                "Count": "$count",
                "_id": 0
            }},
            {"$sort": {"Minute": 1}}
        ]

        result = list(existing_collection.aggregate(pipeline))
        if result:
            for item in result:
                minute = item['Minute']
                time_range = f"{hour+2:02}:{minute:02}"  # Construct time range in HH:MM format
                gender = item['Gender']
                count = item['Count']

                data.append({
                    'Time Range': time_range,
                    'Gender': gender,
                    'Count': count,
                })

            # Create DataFrame from collected data
            df = pd.DataFrame(data)
            # Pivot DataFrame to get male and female counts separately
            df_pivot = df.pivot_table(index='Time Range', columns='Gender', values='Count', aggfunc='sum',
                                    fill_value=0)

            # Add a total count column
            # df_pivot['Total Count'] = df_pivot['Female'] + df_pivot['Male']
            # Ensure all minutes in the hour are covered, and fill in missing minutes with zero counts
            all_minutes = [f"{hour+2:02}:{minute:02}" for minute in range(60)]
            df_pivot = df_pivot.reindex(all_minutes, fill_value=0)
            df_pivot.columns.name = None
            df_pivot.reset_index(inplace=True)
                
            try :                                        
                        try :
                        # Add a total count column
                            df_pivot['Total Count'] = df_pivot['Female'] + df_pivot['Male']
                            df_pivot['Camera Name'] = CameraName   
                        except : 
                            df_pivot['Total Count'] = df_pivot['Female'] 
                            df_pivot['Female'] = 0                        
                            df_pivot['Camera Name'] = CameraName                   
            except :
                            df_pivot['Total Count'] = df_pivot['Male'] 
                            df_pivot['Female'] = 0                        

                            df_pivot['Camera Name'] = CameraName              
                        

    if len(data) == 0:
                dictionary = {'Camera Name': CameraName, 'Time Range': TargetDate,
                              'Female' : 0 ,'Male' :0, 'Total Count': 0}
                return dictionary
    else :
            dictionary = df_pivot.to_dict(orient='records')
            return dictionary
        

# # Example usage:
# hour_data = gender_filtering_Minutes_aggregates('GenderTest', 13, 25,3, 2024)
# pprint.pprint(hour_data)


def get_all_cameras_genderPerMinutes(hour, day, month, year):
    """
    Filter data by date and hour, and get the count for male, female, and total count for each minute within that hour.

    Args:
        hour (int): Hour for which data is to be aggregated.
        day (int): Day component of the date.
        month (int): Month component of the date.
        year (int): Year component of the date.

    Returns:
        pandas.DataFrame: DataFrame containing minute-wise count for male, female, and total count for the given hour.
    """
    # Ensure month, day, and hour are zero-padded if less than 10
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)
    hour = int(hour) 
    hour -=2
    # Construct target date string
    TargetDate = f"{year}-{month_str}-{day_str}"

    existing_collection = db['ModelGenderData']

    camera_names =  all_cameras_in_model('gender')['Camera Names']
    print(camera_names)
    # camera_names = ['aslom','GenderTest','FristCam']
    all_data = []
    for CameraName in camera_names :
        print('Getting Data of :' , CameraName)
        # Query to filter by camera name and date
        query = {
            'Camera Info.Camera Name': CameraName,
            'Date': TargetDate,
            '$expr': {'$eq': [{'$hour': '$Timestamp'}, hour]}
        }        
        data = []    
        if check_existing_document(existing_collection, query):
            print(f'{CameraName} Camera Found in Gender Collection')

            pipeline = [
                {"$match": query},
                {"$unwind": "$Label"},
                {"$group": {
                    "_id": {
                        "Minute": {"$minute": "$Timestamp"},
                        "Gender": "$Label.Gender"
                    },
                    "count": {"$sum": 1}
                }},
                {"$project": {
                    "Minute": "$_id.Minute",
                    "Gender": "$_id.Gender",
                    "Count": "$count",
                    "_id": 0
                }},
                {"$sort": {"Minute": 1}}
            ]

            result = list(existing_collection.aggregate(pipeline))
            if result:
                for item in result:
                    minute = item['Minute']
                    time_range = f"{hour+2:02}:{minute:02}"  # Construct time range in HH:MM format
                    gender = item['Gender']
                    count = item['Count']

                    data.append({
                        'Time Range': time_range,
                        'Gender': gender,
                        'Count': count,
                    })

                # Create DataFrame from collected data
                df = pd.DataFrame(data)
                # Pivot DataFrame to get male and female counts separately
                df_pivot = df.pivot_table(index='Time Range', columns='Gender', values='Count', aggfunc='sum',
                                        fill_value=0)

                # Add a total count column
                # df_pivot['Total Count'] = df_pivot['Female'] + df_pivot['Male']
                # Ensure all minutes in the hour are covered, and fill in missing minutes with zero counts
                all_minutes = [f"{hour+2:02}:{minute:02}" for minute in range(60)]
                df_pivot = df_pivot.reindex(all_minutes, fill_value=0)
                df_pivot.columns.name = None
                df_pivot.reset_index(inplace=True)
                    
                try :                                        
                            try :
                            # Add a total count column
                                df_pivot['Total Count'] = df_pivot['Female'] + df_pivot['Male']
                                df_pivot['Camera Name'] = CameraName   
                            except : 
                                df_pivot['Total Count'] = df_pivot['Female'] 
                                df_pivot['Female'] = 0                        
                                df_pivot['Camera Name'] = CameraName                   
                except :
                                df_pivot['Total Count'] = df_pivot['Male'] 
                                df_pivot['Female'] = 0                        

                                df_pivot['Camera Name'] = CameraName              
                            

        if len(data) == 0:
                    dictionary = {'Camera Name': CameraName, 'Time Range': TargetDate,
                                'Female' : 0 ,'Male' :0, 'Total Count': 0}
                    # return dictionary
        else :
                dictionary = df_pivot.to_dict(orient='records')
                all_data.append(dictionary)
    return all_data    

# pprint.pprint(get_all_cameras_genderPerMinutes(13,25,3,2024))


# camera_names =  all_cameras_in_model('gender')['Camera Names']
# print(camera_names)


def postcam_getallmodelsperSecond(CameraName, minute, hour, day, month, year):
    """
    Filter data by date and hour and get all count data for all models applied to a specific camera, grouped by seconds within the specified minute.

    Args:
        CameraName (str): Name of the camera.
        hour (int): Hour component of the time range.
        minute (int): Minute component of the time range.
        day (int): Day component of the date.
        month (int): Month component of the date.
        year (int): Year component of the date.

    Returns:
        list of dicts: List containing dictionaries with 'Camera Name', 'Time Range', 'Model', 'Count', and 'Second' keys.
    """
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)

    TargetDate = f"{year}-{month_str}-{day_str}"

    model_collection_mapping = {
        'crossingBorder': 'ModelCountingData',
        'crowdedDensity': 'ModelDensityData',
        'crowded': 'ModelCrowdedData',
    }

    hour = int(hour)
    hour = hour - 2
    
    minute = int(minute)

    data = []
    try:
        for ModelName, collection_name in model_collection_mapping.items():
            existing_collection = db[collection_name]

            query = {
                'Camera Info.Camera Name': CameraName,
                'Date': TargetDate,
                '$expr': {'$and': [
                    {'$eq': [{'$hour': '$Timestamp'}, hour]},
                    {'$eq': [{'$minute': '$Timestamp'}, minute]},
                    {'$gte': [{'$second': '$Timestamp'}, 0]},
                    {'$lt': [{'$second': '$Timestamp'}, 60]}  # Assuming seconds range from 0 to 59
                ]}
            }

            if check_existing_document(existing_collection, query):
                print(f'{CameraName} Camera Found in {ModelName} Collection')
                pipeline = [
                    {"$match": query},
                    {"$project": {
                        "Second": {"$second": "$Timestamp"},
                        "Count": "$Count",
                        "_id": 0
                    }}
                ]

                result = list(existing_collection.aggregate(pipeline))
                print(result)
                if result:
                    for item in result:
                        second = item['Second']
                        count = item['Count']
                        time_range = f"{hour+2:02}:{minute:02}:{second:02}"
                        data.append({'Camera Name': CameraName, 'Time Range': time_range, 'Model': ModelName, 'Count': count, 'Second': second})

                break

        if len(data) == 0:
            dictionary = {'Camera Name': CameraName, 'Time Range': f"{hour+2:02}:{minute:02}", 'Count': 0, 'Model': 'Not Found', 'Second': -1}
            return [dictionary]
        
            # for second in range(60):
            #     time_range = f"{hour+2:02}:{minute:02}:{second:02}"
            #     data.append({'Camera Name': CameraName, 'Time Range': time_range, 'Model': 'Not Found', 'Count': 0, 'Second': second})

            
            # return data
        
        else:
            return data

    except Exception as e:
        print(e)
        dictionary = {'Camera Name': CameraName, 'Time Range': f"{hour+2:02}:{minute:02}", 'Count': 0, 'Model': 'Not Found', 'Second': -1}
        return [dictionary]

    
def postcam_getvechileSeconds(CameraName, minute, hour, day, month, year):
    """
    Filter data by date and hour and get all count data for all models applied to a specific camera, grouped by seconds within the specified minute.

    Args:
        CameraName (str): Name of the camera.
        hour (int): Hour component of the time range.
        minute (int): Minute component of the time range.
        day (int): Day component of the date.
        month (int): Month component of the date.
        year (int): Year component of the date.

    Returns:
        list of dicts: List containing dictionaries with 'Camera Name', 'Time Range', 'Model', 'Count', and 'Second' keys.
    """
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)

    TargetDate = f"{year}-{month_str}-{day_str}"


    model_collection_mapping = {
        'vehicle': 'ModelVehicleData',
    }
    hour = int(hour)
    hour = hour - 2
    
    minute = int(minute)

    data = []
    try:
        for ModelName, collection_name in model_collection_mapping.items():
            existing_collection = db[collection_name]

            query = {
                'Camera Info.Camera Name': CameraName,
                'Date': TargetDate,
                '$expr': {'$and': [
                    {'$eq': [{'$hour': '$Timestamp'}, hour]},
                    {'$eq': [{'$minute': '$Timestamp'}, minute]},
                    {'$gte': [{'$second': '$Timestamp'}, 0]},
                    {'$lt': [{'$second': '$Timestamp'}, 60]}  # Assuming seconds range from 0 to 59
                ]}
            }

            if check_existing_document(existing_collection, query):
                print(f'{CameraName} Camera Found in {ModelName} Collection')
                pipeline = [
                    {"$match": query},
                    {"$project": {
                        "Second": {"$second": "$Timestamp"},
                        "Count": "$Count",
                        "_id": 0
                    }}
                ]

                result = list(existing_collection.aggregate(pipeline))
                print(result)
                if result:
                    for item in result:
                        second = item['Second']
                        count = item['Count']
                        time_range = f"{hour+2:02}:{minute:02}:{second:02}"
                        data.append({'Camera Name': CameraName, 'Time Range': time_range, 'Model': ModelName, 'Count': count, 'Second': second})

                break

        if len(data) == 0:
            dictionary = {'Camera Name': CameraName, 'Time Range': f"{hour+2:02}:{minute:02}", 'Count': 0, 'Model': 'Not Found', 'Second': -1}
            return [dictionary]
        
            # for second in range(60):
            #     time_range = f"{hour+2:02}:{minute:02}:{second:02}"
            #     data.append({'Camera Name': CameraName, 'Time Range': time_range, 'Model': 'Not Found', 'Count': 0, 'Second': second})

            
            # return data
        
        else:
            return data

    except Exception as e:
        print(e)
        dictionary = {'Camera Name': CameraName, 'Time Range': f"{hour+2:02}:{minute:02}", 'Count': 0, 'Model': 'Not Found', 'Second': -1}
        return [dictionary]

# pprint.pprint(postcam_getvechileSeconds('MinutesTesting',8,12,25,3,2024))

# pprint.pprint(postcam_getallmodelsperMinutes('MinutesTesting',11,25,3,2024))

def get_all_cameras_count_perSecond(minute, hour, day, month, year):
    """
    Filter data by date and hour and get all count data for all models applied to a specific camera, grouped by seconds within the specified minute.

    Args:
        CameraName (str): Name of the camera.
        hour (int): Hour component of the time range.
        minute (int): Minute component of the time range.
        day (int): Day component of the date.
        month (int): Month component of the date.
        year (int): Year component of the date.

    Returns:
        list of dicts: List containing dictionaries with 'Camera Name', 'Time Range', 'Model', 'Count', and 'Second' keys.
    """
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)

    TargetDate = f"{year}-{month_str}-{day_str}"

    model_collection_mapping = {
        'crossingBorder': 'ModelCountingData',
        'crowdedDensity': 'ModelDensityData',
        'crowded': 'ModelCrowdedData',
    }

    hour = int(hour)
    hour = hour - 2
    
    minute = int(minute)

    all_data = []
    camera_names = finding_camera_names()
    # camera_names = ['aslom', 'MinutesTesting']    
     
    try:
        for CameraName in camera_names:
            print('Getting Data of :' , CameraName)
            data = []  # Initialize data list for each camera           
            for ModelName, collection_name in model_collection_mapping.items():
                existing_collection = db[collection_name]

                query = {
                    'Camera Info.Camera Name': CameraName,
                    'Date': TargetDate,
                    '$expr': {'$and': [
                        {'$eq': [{'$hour': '$Timestamp'}, hour]},
                        {'$eq': [{'$minute': '$Timestamp'}, minute]},
                        {'$gte': [{'$second': '$Timestamp'}, 0]},
                        {'$lt': [{'$second': '$Timestamp'}, 60]}  # Assuming seconds range from 0 to 59
                    ]}
                }

                if check_existing_document(existing_collection, query):
                    print(f'{CameraName} Camera Found in {ModelName} Collection')
                    pipeline = [
                        {"$match": query},
                        {"$project": {
                            "Second": {"$second": "$Timestamp"},
                            "Count": "$Count",
                            "_id": 0
                        }}
                    ]

                    result = list(existing_collection.aggregate(pipeline))
                    print(result)
                    if result:
                        for item in result:
                            second = item['Second']
                            count = item['Count']
                            time_range = f"{hour+2:02}:{minute:02}:{second:02}"
                            data.append({'Camera Name': CameraName, 'Time Range': time_range, 'Model': ModelName, 'Count': count, 'Second': second})

                    break

            if len(data) == 0:
                dictionary = {'Camera Name': CameraName, 'Time Range': f"{hour+2:02}:{minute:02}", 'Count': 0, 'Model': 'Not Found', 'Second': -1}
                # return [dictionary]
            
                # for second in range(60):
                #     time_range = f"{hour+2:02}:{minute:02}:{second:02}"
                #     data.append({'Camera Name': CameraName, 'Time Range': time_range, 'Model': 'Not Found', 'Count': 0, 'Second': second})

                
                # return data
            
            else:
                all_data.append(data)
        return all_data
    except Exception as e:
        print(e)
        dictionary = {'Camera Name': CameraName, 'Time Range': f"{hour+2:02}:{minute:02}", 'Count': 0, 'Model': 'Not Found', 'Second': -1}
        return [[dictionary]]

# pprint.pprint(get_all_cameras_count_perSecond(19,11,25,3,2024))
    
def get_all_vechile_count_perSecond(minute, hour, day, month, year):
    """
    Filter data by date and hour and get all count data for all models applied to a specific camera, grouped by seconds within the specified minute.

    Args:
        CameraName (str): Name of the camera.
        hour (int): Hour component of the time range.
        minute (int): Minute component of the time range.
        day (int): Day component of the date.
        month (int): Month component of the date.
        year (int): Year component of the date.

    Returns:
        list of dicts: List containing dictionaries with 'Camera Name', 'Time Range', 'Model', 'Count', and 'Second' keys.
    """
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)

    TargetDate = f"{year}-{month_str}-{day_str}"

    model_collection_mapping = {
        'vehicle': 'ModelVehicleData',
    }

    hour = int(hour)
    hour = hour - 2
    
    minute = int(minute)

    all_data = []
    camera_names = finding_camera_names()
    # camera_names = ['aslom', 'MinutesTesting']    
     
    try:
        for CameraName in camera_names:
            print('Getting Data of :' , CameraName)
            data = []  # Initialize data list for each camera           
            for ModelName, collection_name in model_collection_mapping.items():
                existing_collection = db[collection_name]

                query = {
                    'Camera Info.Camera Name': CameraName,
                    'Date': TargetDate,
                    '$expr': {'$and': [
                        {'$eq': [{'$hour': '$Timestamp'}, hour]},
                        {'$eq': [{'$minute': '$Timestamp'}, minute]},
                        {'$gte': [{'$second': '$Timestamp'}, 0]},
                        {'$lt': [{'$second': '$Timestamp'}, 60]}  # Assuming seconds range from 0 to 59
                    ]}
                }

                if check_existing_document(existing_collection, query):
                    print(f'{CameraName} Camera Found in {ModelName} Collection')
                    pipeline = [
                        {"$match": query},
                        {"$project": {
                            "Second": {"$second": "$Timestamp"},
                            "Count": "$Count",
                            "_id": 0
                        }}
                    ]

                    result = list(existing_collection.aggregate(pipeline))
                    print(result)
                    if result:
                        for item in result:
                            second = item['Second']
                            count = item['Count']
                            time_range = f"{hour+2:02}:{minute:02}:{second:02}"
                            data.append({'Camera Name': CameraName, 'Time Range': time_range, 'Model': ModelName, 'Count': count, 'Second': second})

                    break

            if len(data) == 0:
                dictionary = {'Camera Name': CameraName, 'Time Range': f"{hour+2:02}:{minute:02}", 'Count': 0, 'Model': 'Not Found', 'Second': -1}
                # return [dictionary]
            
                # for second in range(60):
                #     time_range = f"{hour+2:02}:{minute:02}:{second:02}"
                #     data.append({'Camera Name': CameraName, 'Time Range': time_range, 'Model': 'Not Found', 'Count': 0, 'Second': second})

                
                # return data
            
            else:
                all_data.append(data)
        return all_data
    except Exception as e:
        print(e)
        dictionary = {'Camera Name': CameraName, 'Time Range': f"{hour+2:02}:{minute:02}", 'Count': 0, 'Model': 'Not Found', 'Second': -1}
        return [[dictionary]]
    
# pprint.pprint(get_all_vechile_count_perSecond(8,12,25,3,2024))


def gender_filtering_Seconds_aggregates(CameraName, minute,hour, day, month, year):
    """
    Filter data by date and hour, and get the sum for male, female, and total count for each second within the specified minute.

    Args:
        CameraName (str): Name of the camera.
        hour (int): Hour component of the time range.
        minute (int): Minute component of the time range.
        day (int): Day component of the date.
        month (int): Month component of the date.
        year (int): Year component of the date.

    Returns:
        pandas.DataFrame: DataFrame containing second-wise count for male, female, and total count for the given minute.
    """
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)
    
    TargetDate = f"{year}-{month_str}-{day_str}"

    existing_collection = db['ModelGenderData']

    hour = int(hour)
    minute = int(minute)
    hour = hour -2
    query = {
        'Camera Info.Camera Name': CameraName,
        'Date': TargetDate,
        '$expr': {'$and': [
            {'$eq': [{'$hour': '$Timestamp'}, hour]},
            {'$eq': [{'$minute': '$Timestamp'}, minute]},
            {'$gte': [{'$second': '$Timestamp'}, 0]},
            {'$lt': [{'$second': '$Timestamp'}, 60]}  # Assuming seconds range from 0 to 59
        ]}
    }

    data = []
    if check_existing_document(existing_collection, query):
        print(f'{CameraName} Camera Found in Gender Collection')

        pipeline = [
            {"$match": query},
            {"$unwind": "$Label"},
            {"$group": {
                "_id": {
                    "Second": {"$second": "$Timestamp"},
                    "Gender": "$Label.Gender"
                },
                "count": {"$sum": 1}
            }},
            {"$project": {
                "Second": "$_id.Second",
                "Gender": "$_id.Gender",
                "Count": "$count",
                "_id": 0
            }},
            {"$sort": {"Second": 1}}
        ]

        result = list(existing_collection.aggregate(pipeline))
        if result:
            for item in result:
                second = item['Second']
                gender = item['Gender']
                count = item['Count']

                data.append({
                    'Time Range': f"{hour+2:02}:{minute:02}:{second:02}",
                    'Gender': gender,
                    'Count': count,
                    'Second': second
                })
        # Create DataFrame from collected data
        df = pd.DataFrame(data)
        # Pivot DataFrame to get male and female counts    
        df_pivot = df.pivot_table(index='Second', columns='Gender', values='Count', aggfunc='sum', fill_value=0)
        # Add a total count column
        # df_pivot['Total Count'] = df_pivot['Female'] + df_pivot['Male']
        # Ensure all seconds in the minute are covered, and fill in missing seconds with zero counts
        # all_seconds = [second for second in range(60)]
        # df_pivot = df_pivot.reindex(all_seconds, fill_value=0)
        df_pivot.columns.name = None
        df_pivot.reset_index(inplace=True)                
        try :                                        
                try :
                        # Add a total count column
                            df_pivot['Total Count'] = df_pivot['Female'] + df_pivot['Male']
                            df_pivot['Camera Name'] = CameraName   
                except : 
                            df_pivot['Total Count'] = df_pivot['Female'] 
                            df_pivot['Female'] = 0                        
                            df_pivot['Camera Name'] = CameraName                   
        except :
                            df_pivot['Total Count'] = df_pivot['Male'] 
                            df_pivot['Female'] = 0                        
                            df_pivot['Camera Name'] = CameraName     
        df_pivot['Time Range'] = f"{hour+2:02}:{minute:02}:" + df_pivot['Second'].astype(str).str.zfill(2)

    if len(data) == 0:
        # If no data found for the given minute, return a dictionary with zero counts for all seconds
        dictionary = [{'Camera Name': CameraName, 'Time Range': f"{hour:02}:{minute:02}:{second:02}", 'Female': 0, 'Male': 0, 'Total Count': 0, 'Second': second} for second in range(60)]
        return dictionary
    else:
        dictionary = df_pivot.to_dict(orient='records')
        return dictionary

# hour_data = gender_filtering_Seconds_aggregates('GenderTest', 13,8, 25,3, 2024)
# pprint.pprint(hour_data)


def get_all_cameras_genderPerSeconds(minute, hour, day, month, year):
    """
    Filter data by date and hour, and get the sum for male, female, and total count for each second within the specified minute.

    Args:
        CameraName (str): Name of the camera.
        hour (int): Hour component of the time range.
        minute (int): Minute component of the time range.
        day (int): Day component of the date.
        month (int): Month component of the date.
        year (int): Year component of the date.

    Returns:
        pandas.DataFrame: DataFrame containing second-wise count for male, female, and total count for the given minute.
    """
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)
    
    TargetDate = f"{year}-{month_str}-{day_str}"

    existing_collection = db['ModelGenderData']

    hour = int(hour)
    minute = int(minute)
    hour = hour -2
    camera_names =  all_cameras_in_model('gender')['Camera Names']
    print(camera_names)
    # camera_names = ['aslom','GenderTest','FristCam']
    all_data = []
    for CameraName in camera_names :
        print('Getting Data of :' , CameraName)
        # Query to filter by camera name and date
        query = {
            'Camera Info.Camera Name': CameraName,
            'Date': TargetDate,
            '$expr': {'$eq': [{'$hour': '$Timestamp'}, hour]}
        }        
        data = []        

        if check_existing_document(existing_collection, query):
            print(f'{CameraName} Camera Found in Gender Collection')

            pipeline = [
                {"$match": query},
                {"$unwind": "$Label"},
                {"$group": {
                    "_id": {
                        "Second": {"$second": "$Timestamp"},
                        "Gender": "$Label.Gender"
                    },
                    "count": {"$sum": 1}
                }},
                {"$project": {
                    "Second": "$_id.Second",
                    "Gender": "$_id.Gender",
                    "Count": "$count",
                    "_id": 0
                }},
                {"$sort": {"Second": 1}}
            ]

            result = list(existing_collection.aggregate(pipeline))
            if result:
                for item in result:
                    second = item['Second']
                    gender = item['Gender']
                    count = item['Count']

                    data.append({
                        'Time Range': f"{hour+2:02}:{minute:02}:{second:02}",
                        'Gender': gender,
                        'Count': count,
                        'Second': second
                    })
            # Create DataFrame from collected data
            df = pd.DataFrame(data)
            # Pivot DataFrame to get male and female counts    
            df_pivot = df.pivot_table(index='Second', columns='Gender', values='Count', aggfunc='sum', fill_value=0)
            # Add a total count column
            # df_pivot['Total Count'] = df_pivot['Female'] + df_pivot['Male']
            # Ensure all seconds in the minute are covered, and fill in missing seconds with zero counts
            # all_seconds = [second for second in range(60)]
            # df_pivot = df_pivot.reindex(all_seconds, fill_value=0)
            df_pivot.columns.name = None
            df_pivot.reset_index(inplace=True)                
            try :                                        
                    try :
                            # Add a total count column
                                df_pivot['Total Count'] = df_pivot['Female'] + df_pivot['Male']
                                df_pivot['Camera Name'] = CameraName   
                    except : 
                                df_pivot['Total Count'] = df_pivot['Female'] 
                                df_pivot['Female'] = 0                        
                                df_pivot['Camera Name'] = CameraName                   
            except :
                                df_pivot['Total Count'] = df_pivot['Male'] 
                                df_pivot['Female'] = 0                        
                                df_pivot['Camera Name'] = CameraName 
            df_pivot['Time Range'] = f"{hour+2:02}:{minute:02}:" + df_pivot['Second'].astype(str).str.zfill(2)
                                                
        if len(data) == 0:
            # If no data found for the given minute, return a dictionary with zero counts for all seconds
            dictionary = [{'Camera Name': CameraName, 'Time Range': f"{hour:02}:{minute:02}:{second:02}", 'Female': 0, 'Male': 0, 'Total Count': 0, 'Second': second} for second in range(60)]
            # return dictionary
        else:
            dictionary = df_pivot.to_dict(orient='records')
            all_data.append(dictionary)

    return    all_data      


# hour_data = get_all_cameras_genderPerSeconds(13,8, 25,3, 2024)
# pprint.pprint(hour_data)
###########################################################
def Age_filtering_Seconds_aggregates(CameraName, minute,hour, day, month, year):
    """
    Filter data by date and hour, and get the sum for old,young and total count for each second within the specified minute.

    Args:
        CameraName (str): Name of the camera.
        hour (int): Hour component of the time range.
        minute (int): Minute component of the time range.
        day (int): Day component of the date.
        month (int): Month component of the date.
        year (int): Year component of the date.

    Returns:
        pandas.DataFrame: DataFrame containing second-wise count for old,young and total count for the given minute.
    """
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)
    
    TargetDate = f"{year}-{month_str}-{day_str}"

    existing_collection = db['ModelAgeData']

    hour = int(hour)
    minute = int(minute)
    hour = hour -2
    query = {
        'Camera Info.Camera Name': CameraName,
        'Date': TargetDate,
        '$expr': {'$and': [
            {'$eq': [{'$hour': '$Timestamp'}, hour]},
            {'$eq': [{'$minute': '$Timestamp'}, minute]},
            {'$gte': [{'$second': '$Timestamp'}, 0]},
            {'$lt': [{'$second': '$Timestamp'}, 60]}  # Assuming seconds range from 0 to 59
        ]}
    }

    data = []
    if check_existing_document(existing_collection, query):
        print(f'{CameraName} Camera Found in Age Collection')

        pipeline = [
            {"$match": query},
            {"$unwind": "$Label"},
            {"$group": {
                "_id": {
                    "Second": {"$second": "$Timestamp"},
                    "Age": "$Label.Age"
                },
                "count": {"$sum": 1}
            }},
            {"$project": {
                "Second": "$_id.Second",
                "Age": "$_id.Age",
                "Count": "$count",
                "_id": 0
            }},
            {"$sort": {"Second": 1}}
        ]

        result = list(existing_collection.aggregate(pipeline))
        if result:
            for item in result:
                second = item['Second']
                Age = item['Age']
                count = item['Count']

                data.append({
                    'Time Range': f"{hour+2:02}:{minute:02}:{second:02}",
                    'Age': Age,
                    'Count': count,
                    'Second': second
                })
        # Create DataFrame from collected data
        df = pd.DataFrame(data)
        # Pivot DataFrame to get male and female counts    
        df_pivot = df.pivot_table(index='Second', columns='Age', values='Count', aggfunc='sum', fill_value=0)
        # Add a total count column
        # Ensure all seconds in the minute are covered, and fill in missing seconds with zero counts
        # all_seconds = [second for second in range(60)]
        # df_pivot = df_pivot.reindex(all_seconds, fill_value=0)
        df_pivot.columns.name = None
        df_pivot.reset_index(inplace=True)                
        try :                                        
                try :
                        # Add a total count column
                            df_pivot['Total Count'] = df_pivot['Old'] + df_pivot['Young']
                            df_pivot['Camera Name'] = CameraName   
                except : 
                            df_pivot['Total Count'] = df_pivot['Old'] 
                            df_pivot['Old'] = 0                        
                            df_pivot['Camera Name'] = CameraName                   
        except :
                            df_pivot['Total Count'] = df_pivot['Young'] 
                            df_pivot['Young'] = 0                        
                            df_pivot['Camera Name'] = CameraName     
        df_pivot['Time Range'] = f"{hour+2:02}:{minute:02}:" + df_pivot['Second'].astype(str).str.zfill(2)

    if len(data) == 0:
        # If no data found for the given minute, return a dictionary with zero counts for all seconds
        dictionary = [{'Camera Name': CameraName, 'Time Range': f"{hour:02}:{minute:02}:{second:02}", 'Old': 0, 'Young': 0, 'Total Count': 0, 'Second': second} for second in range(60)]
        return dictionary
    else:
        dictionary = df_pivot.to_dict(orient='records')
        return dictionary



def Age_filtering_Minutes_aggregates(CameraName, hour, day, month, year):
    """
    Filter data by date and hour, and get the count for Old,Young and total count for each minute within that hour.

    Args:
        CameraName (str): Name of the camera.
        hour (int): Hour for which data is to be aggregated.
        day (int): Day component of the date.
        month (int): Month component of the date.
        year (int): Year component of the date.

    Returns:
        pandas.DataFrame: DataFrame containing minute-wise count for Old,Young, and total count for the given hour.
    """
    # Ensure month, day, and hour are zero-padded if less than 10
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)
    hour = int(hour) 
    hour -=2
    # Construct target date string
    TargetDate = f"{year}-{month_str}-{day_str}"

    existing_collection = db['ModelAgeData']

    # Query to filter by camera name, date, and hour
    query = {
        'Camera Info.Camera Name': CameraName,
        'Date': TargetDate,
        '$expr': {'$eq': [{'$hour': '$Timestamp'}, hour]}
    }

    data = []
    if check_existing_document(existing_collection, query):
        print(f'{CameraName} Camera Found in Age Collection')

        pipeline = [
            {"$match": query},
            {"$unwind": "$Label"},
            {"$group": {
                "_id": {
                    "Minute": {"$minute": "$Timestamp"},
                    "Age": "$Label.Age"
                },
                "count": {"$sum": 1}
            }},
            {"$project": {
                "Minute": "$_id.Minute",
                "Age": "$_id.Age",
                "Count": "$count",
                "_id": 0
            }},
            {"$sort": {"Minute": 1}}
        ]

        result = list(existing_collection.aggregate(pipeline))
        if result:
            for item in result:
                minute = item['Minute']
                time_range = f"{hour+2:02}:{minute:02}"  # Construct time range in HH:MM format
                Age = item['Age']
                count = item['Count']

                data.append({
                    'Time Range': time_range,
                    'Age': Age,
                    'Count': count,
                })

            # Create DataFrame from collected data
            df = pd.DataFrame(data)
            # Pivot DataFrame to get male and female counts separately
            df_pivot = df.pivot_table(index='Time Range', columns='Age', values='Count', aggfunc='sum',
                                    fill_value=0)

            # Add a total count column
            # Ensure all minutes in the hour are covered, and fill in missing minutes with zero counts
            all_minutes = [f"{hour+2:02}:{minute:02}" for minute in range(60)]
            df_pivot = df_pivot.reindex(all_minutes, fill_value=0)
            df_pivot.columns.name = None
            df_pivot.reset_index(inplace=True)
                
            try :                                        
                        try :
                        # Add a total count column
                            df_pivot['Total Count'] = df_pivot['Old'] + df_pivot['Young']
                            df_pivot['Camera Name'] = CameraName   
                        except : 
                            df_pivot['Total Count'] = df_pivot['Old'] 
                            df_pivot['Old'] = 0                        
                            df_pivot['Camera Name'] = CameraName                   
            except :
                            df_pivot['Total Count'] = df_pivot['Young'] 
                            df_pivot['Young'] = 0                        

                            df_pivot['Camera Name'] = CameraName              
                        

    if len(data) == 0:
                dictionary = {'Camera Name': CameraName, 'Time Range': TargetDate,
                              'Old' : 0 ,'Young' :0, 'Total Count': 0}
                return dictionary
    else :
            dictionary = df_pivot.to_dict(orient='records')
            return dictionary

def Age_filtering_date_aggrigates(CameraName, day, month, year):
    
    """
    Filter data by date and get the average count in the form of time range.

    Args:
        CameraName (str): Name of the camera.
        day (int): Day component of the date.
        month (int): Month component of the date.
        year (int): Year component of the date.

    Returns:
        pandas.DataFrame: DataFrame containing time range, total count for old, and total count for young.
    """
    # Ensure month and day are zero-padded if less than 10
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)

    # Construct target date string
    TargetDate = f"{year}-{month_str}-{day_str}"

    existing_collection = db['ModelAgeData']
    
    # Query to filter by camera name and date
    query = {'Camera Info.Camera Name': CameraName, 'Date': TargetDate}
    data = []
    if check_existing_document(existing_collection, query):
        print(f'{CameraName} Camera Found in Age Collection')
        
        pipeline = [
            {"$match": query},
            {"$unwind": "$Label"},
            {"$group": {
                "_id": {
                    "Hour": {"$hour": "$Timestamp"},
                    "Age": "$Label.Age"
                },
                "count": {"$sum": 1}
            }},
            {"$project": {
                "Hour": "$_id.Hour",
                "Age": "$_id.Age",
                "Count": "$count",
                "_id": 0
            }},
            {"$sort": {"Hour": 1}}
        ]


        result = list(existing_collection.aggregate(pipeline))
        if result:
            
            for item in result:
                hour = item['Hour']
                Age = item['Age']
                count = item['Count']
                
                # Adjusting hour for display
                am_pm = "PM" if (hour + 2) >= 12 else "AM"
                formatted_hour = (hour + 2) if (hour + 2) <= 12 else (hour + 2) - 12
                time_range = f"{formatted_hour} {am_pm} - {formatted_hour + 1} {am_pm}"
                
                data.append({
                    'Time Range': time_range,
                    'Age': Age,
                    'Count': count,
                })

                for hour in range(24):
                        am_pm = "PM" if (hour) >= 12 else "AM"
                        formatted_hour = (hour) if (hour) <= 12 else (hour) - 12
                        time_range = f"{formatted_hour} {am_pm} - {formatted_hour + 1} {am_pm}"
                
                        data.append({
                            'Time Range': time_range,
                            'Age': Age,
                            'Count': 0,
                        })

                        
            
            # Create DataFrame from collected data
            df = pd.DataFrame(data)
            
            # Pivot DataFrame to get male and female counts separately
            df_pivot = df.pivot_table(index='Time Range', columns='Age', values='Count', aggfunc='sum', fill_value=0)
            
            # Reset index to make 'Time Range' a column
            df_pivot.reset_index(inplace=True)
            
            # Rename columns for clarity
            df_pivot.columns.name = None
            
            try :                                        
                    try :
                    # Add a total count column
                        df_pivot['Total Count'] = df_pivot['Old'] + df_pivot['Young']
                        df_pivot['Camera Name'] = CameraName   
                    except : 
                        df_pivot['Total Count'] = df_pivot['Old'] 
                        df_pivot['Old'] = 0                        
                        df_pivot['Camera Name'] = CameraName                   
            except :
                        df_pivot['Total Count'] = df_pivot['Young'] 
                        df_pivot['Young'] = 0                        

                        df_pivot['Camera Name'] = CameraName              
            

    if len(data) == 0:
                dictionary = {'Camera Name': CameraName, 'Time Range': TargetDate,
                              'Old' : 0 ,'Young' :0, 'Total Count': 0}
                return dictionary
    else :

            # Define the order of time ranges
            df = df_pivot.sort_values(by=['Time Range', 'Total Count'], ascending=[True, False]).drop_duplicates(subset='Time Range')
            # Define the order of time ranges
            time_range_order = [f"{i} AM - {i + 1} AM" if i != 12 else f"{i} AM - {i + 1} PM" for i in range(12)]
            time_range_order.extend([f"{i} PM - {i + 1} PM" if i != 12 else f"{i} PM - {i + 1} AM" for i in range(12)])
            ind =  time_range_order.index('0 PM - 1 PM')
            time_range_order[ind] = '12 PM - 1 PM'

            # Convert 'Time Range' column to categorical with predefined order
            df['Time Range'] = pd.Categorical(df['Time Range'], categories=time_range_order, ordered=True)

            # Sort DataFrame by 'Time Range'
            df.sort_values(by='Time Range', inplace=True)
            df.dropna(inplace=True)
            return df          
  
def Age_filtering_month_aggregates(CameraName, month, year):

    """
    Filter data by month and get the total count for old and young for each day.

    Args:
        CameraName (str): Name of the camera.
        month (int): Month component of the date.
        year (int): Year component of the date.

    Returns:
        pandas.DataFrame: DataFrame containing total count for old and young for each day.
    """
    # Ensure month is zero-padded if less than 10
    month = int(month)
    year = int(year)
    month_str = str(month).zfill(2)
    # Get the number of days in the month
    num_days = calendar.monthrange(year, month)[1]

    # Initialize an empty list to store data for all days in the month
    data = []

    # Iterate over all days in the month
    for day in range(1, num_days + 1):
        # Retrieve data for the specific day using the existing function
        daily_data = gender_filtering_date_aggrigates(CameraName, day, month, year)
        
        # If there's no data for the day, add zeros for old and young counts
        if isinstance(daily_data, dict): # If there's no data for the day
            data.append({
                'Camera Name': CameraName,
                'Time Range': f"{year}-{month_str}-{str(day).zfill(2)}",
                'Old': 0,
                'Young': 0,
                'Total Count': 0
            })
        else:
            # Calculate the total count for old and young for the day
            Old_count = daily_data['Old'].sum()
            Young_count = daily_data['Young'].sum()
            total_count = Old_count + Young_count
            
            # Add the data for the day to the list
            data.append({
                'Camera Name': CameraName,
                'Time Range': f"{year}-{month_str}-{str(day).zfill(2)}",
                'Old': Old_count,
                'Young': Young_count,
                'Total Count': total_count
            })

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)

    # Return the DataFrame
    return df     

def Age_filtering_year_aggregates(CameraName, year):
    """
    Filter data by year and get the total count for old and young for each month.

    Args:
        CameraName (str): Name of the camera.
        year (int): Year component.

    Returns:
        pandas.DataFrame: DataFrame containing total count for old and young for each month.
    """
    # Initialize an empty list to store data for all months in the year
    data = []
    year = int(year)

    # Iterate over all months in the year
    for month in range(1, 13):
        # Get the number of days in the month
        num_days = calendar.monthrange(year, month)[1]

        # Initialize counters for old and young counts for the month
        old_count_month = 0
        young_count_month = 0

        # Iterate over all days in the month
        for day in range(1, num_days + 1):
            # Retrieve data for the specific day using the existing function
            daily_data = gender_filtering_date_aggrigates(CameraName, day, month, year)
            
            # If there's data for the day, sum the male and female counts
            if not isinstance(daily_data, dict): # If there's no data for the day
                old_count_month += daily_data['Old'].sum()
                young_count_month += daily_data['Young'].sum()

        # Calculate the total count for the month
        total_count_month = old_count_month + young_count_month
        
        # Add the aggregated data for the month to the list
        data.append({
            'Camera Name': CameraName,
            'Time Range': calendar.month_name[int(month)],
            'Year' : year ,
            'Young': young_count_month,
            'Old': old_count_month,
            'Total Count': total_count_month
        })

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)

    # Return the DataFrame
    return df

             


