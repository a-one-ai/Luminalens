from pymongo import MongoClient
from datetime import datetime
import pandas as pd
import pytz
import math
from pymongo.errors import OperationFailure
from time import sleep
import calendar

# Global variables for client and database connection
global client, db

# Connect to the default local MongoDB instance and assign the client and database as global
client = MongoClient('mongodb://localhost:27017/')
db = client.CameraProject2

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
    else:
        inserted_document = existing_collection.insert_one(data)
        print('Inserted Successfully with ID:', inserted_document.inserted_id)
        return inserted_document

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
    elif ModelName == 'Gender':
        existing_collection = db['ModelGenderData']

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

    if isinstance(Label, int) or ModelName in ['vehicle', 'crowdedDensity', 'crossingBorder', 'crowded']:
        data['Count'] = Label
    elif isinstance(Label, str) or ModelName not in ['vehicle', 'crowdedDensity', 'crossingBorder', 'crowded']:
        data['Label'] = Label

    data['Frame Path'] = FramePath
    data['Timestamp'] = current_time_egypt
    data['Date'] = date_str
    inserted_document = ''
    inserted_document = existing_collection.insert_one(data)
    print(f'Inserted Successfully with ID in {ModelName} Collection: {inserted_document.inserted_id}')
    return inserted_document

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
        'Gender': 'ModelGenderData'
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


# print(postcam_geteachmodelstat('test'))
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
        'crowdedDensity': 'ModelDensityData',
        'crossingBorder': 'ModelCountingData',
        'crowded': 'ModelCrowdedData',
    }

    data = []
    try :
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
                        average_count = item['Count Average']
                        am_pm = "PM" if (hour) >= 12 else "AM"
                        formatted_hour = (hour) if (hour) <= 12 else (hour) - 12
                        time_range = f"{formatted_hour} {am_pm} - {formatted_hour + 1} {am_pm}"
                        
                        data.append({'Model': ModelName, 'Time Range': time_range, 'Count Average': average_count}) 
                        
                    for hour in range(24):
                        am_pm = "PM" if (hour + 2) >= 12 else "AM"
                        formatted_hour = (hour + 2) if (hour + 2) <= 12 else (hour + 2) - 12
                        time_range = f"{formatted_hour} {am_pm} - {formatted_hour + 1} {am_pm}"
                        data.append({'Model': ModelName, 'Time Range': time_range, 'Count Average': 0})                          

                    break                                              
                    

        if len(data) == 0 :
                dictionary = {'Camera Name' :CameraName ,'Time Range' :TargetDate ,'Count Average' :0 , 'Model' : 'Not Found'  }
                return dictionary
        else :
            df =  pd.DataFrame(data)
            df['Camera Name'] = CameraName
            df = df.drop_duplicates(subset='Time Range',keep='first')            
            return df        
    
    except :
        dictionary = {'Camera Name' :CameraName ,'Time Range' : 'Null', 'Count Average' :0 , 'Models' : 'Not Found'  }
        return dictionary
#print(postcam_getallmodelsperH('aslom',19,2,2024))
    
#____________________________________________________________________________    
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
    
#print(postcam_getallmodelsperM('Ro7Elsharq','02','2024'))    
    
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
        return final_df

    
#print(postcam_getallmodelsperY('Ro7Elsharq',2024))

#____________________________________________________________________________
def all_cameras_in_model(ModelName) :
    
    model_collection_mapping = {
        'violence': 'ModelViolenceData',
        'vehicle': 'ModelVehicleData',
        'crowdedDensity': 'ModelDensityData',
        'crossingBorder': 'ModelCountingData',
        'crowded': 'ModelCrowdedData',
        'gender': 'ModelGenderData' }

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
        'gender': 'ModelGenderData' }
    
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
            'gender': 'ModelGenderData' }
        
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
            dictionary['Models'] = 'Not Found'
            return dictionary
        
        else :
            dictionary = dict()
            dictionary['Camera Name'] = CameraName
            dictionary['Camera Info'] = data
            dictionary['Models'] = models_list            
            return dictionary
    else :
        dictionary = dict()
        dictionary['Camera Name'] = CameraName
        dictionary['Camera Info'] ='Not Found'
        dictionary['Models'] = 'Not Found'
        return dictionary
    
#print(all_camera_info('Ro7Elsharq'))