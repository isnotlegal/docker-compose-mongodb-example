from fastapi import FastAPI
from models import Argument, ArgumentResponse
from pymongo import MongoClient
import numpy as np
import pickle
from datetime import datetime

app = FastAPI(title="Titanic API", version="0.1", description="Kader belirleyen servis")

client = MongoClient("mongodb://mongodb:27017/")
db = client.mydatabase
collection = db.mycollection


filename = 'titanic_model.sav'
model = pickle.load(open(filename, 'rb'))

title_mapping = {"Bay": 1, "Hanım": 2, "Bayan": 3, "Usta": 4, "Doktor": 5, "Özgü": 6}
embarked_mapping = {"Southampton, İngiltere": 1, "Cherbourg, Fransa": 2, "Queesntown, İrlanda": 3}
sex_mapping = {"Erkek": 0, "Kadın": 1}

async def prediction(pclass, sex, age, sibsp, parch, embarked, title):
    new_array = np.array([pclass, sex_mapping[sex], age, sibsp, parch, embarked_mapping[embarked], title_mapping[title]]).reshape(1, -1)
    result = model.predict(new_array)
    proba = model.predict_proba(new_array)
    return ("Not Survived", proba[0][0]*100) if result == 0 else ("Survived", proba[0][1]*100)

@app.post('/predict', response_model=ArgumentResponse)
async def predict_survive(arg: Argument):
    results = await prediction(arg.pclass, arg.sex, arg.age, arg.sibsp, arg.parch, arg.embarked, arg.title)
    
    log_entry = {
        "user_input": {
            "pclass": arg.pclass,
            "sex": arg.sex,
            "age": arg.age,
            "sibsp": arg.sibsp,
            "parch": arg.parch,
            "embarked": arg.embarked,
            "title": arg.title
        },
        "predicted_output": results,
        "timestamp": datetime.now()
    }
    collection.insert_one(log_entry)
    print(log_entry)
    return ArgumentResponse(survive=results[0], proba=results[1])
