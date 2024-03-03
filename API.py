from flask import Flask , request , jsonify

app = Flask(__name__)


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



#______________________________________________________
@app.route("/colorDocs" , methods = ['POST'])
def coloeDocs():
    data = request.form 
    if not data :
        return jsonify({"mess" : "No data provided"})
    
    color = data.get('color')
    cameraname = data.get('cameraname')
    docs = function

    return jsonify(docs)



#______________________________________________________
@app.route("/genderCount_Docs" , methods = ['POST'])
def genderCount_Docs():
        data = request.form
        if not data:
            return jsonify({'mess': "No data provided" }) 
        
        cameraName = data.get('cameraname')
        day = data.get('day')
        month = data.get('month')
        year = data.get('year')


        docs = None 

        if  cameraName and day and month and year :
            Male , Female , docs = function(cameraName, day, month, year)
            return jsonify({'Male': Male , "Female" : Female , "Docs" : docs}) 



        elif cameraName and not day and month and year:
            male , female , docs  = function(cameraName, month, year)
            return jsonify({'Male': male , "Female" : female , "Docs" : docs}) 



        elif cameraName and not day and not month and year:
            male , female , docs = function(cameraName, year)
            return jsonify({'Male': male , "Female" : female , "Docs" : docs}) 




#______________________________________________________
@app.route("/violence_Docs" , methods = ['POST'])
def genderCount_Docs():
        data = request.form
        if not data:
            return jsonify({'mess': "No data provided" }) 
        
        cameraName = data.get('cameraname')
        day = data.get('day')
        month = data.get('month')
        year = data.get('year')

 
        docs = None 

        if  cameraName and day and month and year :
            docs = function(cameraName, day, month, year)
            return jsonify({ "Docs" : docs}) 



        elif cameraName and not day and month and year:
            docs  = function(cameraName, month, year)
            return jsonify({ "Docs" : docs}) 



        elif cameraName and not day and not month and year:
            docs = function(cameraName, year)
            return jsonify({ "Docs" : docs}) 

