from typing import Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from utils import *

app = FastAPI()

# Enabling CORS options.
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Defining Api end-points
@app.get('/')
def get_root():
    return {"greetings": "Welcome to Veracious ML model's API"}

@app.get('/validate-fact')
def get_fact_validation(fact: str):
    
    (model, vectorizer) = load_model_vectorizer("../passive_aggressive_model.pkl", "../passive_aggressive_vectorizer.pkl")
    
    pred = fact_validator(fact, model, vectorizer)

    return {"trust_score": pred[0]}