# from functions import *     
from pymongo import MongoClient
from MongoPackageV2 import year_filter_aggerigates_df
from MongoPackageV2 import months_filter_aggerigates_df
from MongoPackageV2 import *


# Global variables for client and database connection


# modelNames = ['crowdedDensity', 'crossingBorder', 'crowded', 'Gender']
# cameraName = "cameraName"

# multiModelRunInsert(cameraName, modelNames)

x = year_filter_aggerigates_df('Ro7Elsharq', 'crowded', '2024')
print(x)

x = months_filter_aggerigates_df('Ro7Elsharq', 'crowded','2', '2024')
print(x)