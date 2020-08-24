from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import numpy as np
import pickle
from flask import jsonify
from pydantic import BaseModel

app = FastAPI()

model = pickle.load(open('model.pkl','rb'))



class inputDetails(BaseModel):
    normalized-losses:int, 
    make:int, 
    num-of-doors:int, 
    body-style:int, 
    drive-wheels:int, 
    engine-location:int, 
    num-of-cylinders:int, 
    engine-size:float, 
    horsepower:float, 
    city-mpg:float, 
    highway-mpg:float,
    diesel:int, 
    gas:int, 
    aspiration-std:int, 
    aspiration-turbo:int

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request
    })

@app.post('/predict')
async def predict(details:inputDetails):
    '''
    For rendering results on HTML GUI
    '''
    print(details.experience)
    print(details.interview_score)
    print(details.test_score)
    int_features = list()
    int_features.append(details.experience)
    int_features.append(details.test_score)
    int_features.append(details.interview_score)
    
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return {"Employee Salary":output}