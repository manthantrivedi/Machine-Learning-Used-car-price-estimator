from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import numpy as np
import pickle
from flask import jsonify
from pydantic import BaseModel

# initialise app
app = FastAPI()

# Read/load model saved in .pkl format
model = pickle.load(open('model.pkl','rb'))

# Define input format
class inputDetails(BaseModel):
    make:int
    body_style:int
    drive_wheels:int 
    horsepower:float 
    city_mpg:float
    highway_mpg:float
    diesel:int
    gas:int
    aspiration_std:int 
    aspiration_turbo:int

@app.get("/")
async def home(request: Request):
    return {"Hello:Home Page"}

@app.post('/predict')
async def predict(details:inputDetails):
    '''
    For rendering results on HTML GUI
    '''
    print(details.make)
    print(details.body_style)
    print(details.drive_wheels)
    print(details.horsepower)
    print(details.city_mpg)
    print(details.highway_mpg)
    print(details.diesel)
    print(details.gas)
    print(details.aspiration_std)
    print(details.aspiration_turbo)

    int_features = list()

    int_features.append(details.make)
    int_features.append(details.body_style)
    int_features.append(details.drive_wheels)
    int_features.append(details.horsepower)
    int_features.append(details.city_mpg)
    int_features.append(details.highway_mpg)
    int_features.append(details.diesel)
    int_features.append(details.gas)
    int_features.append(details.aspiration_std)
    int_features.append(details.aspiration_turbo)
    
    #Convert final list to numpy array datatype for feeding it to the ML model
    final_features = [np.array(int_features)]

    # Make final prediction using the API input Data and feeding it to ML model
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return {"Estimated Price of Car(approx.)":output}