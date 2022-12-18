from typing import Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import numpy as np

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
    
    # PAC Model
    (model, vectorizer) = load_model_vectorizer("../passive_aggressive_model.pkl", "../passive_aggressive_vectorizer.pkl")
    
    pred_pac = fact_validator(fact, model, vectorizer)

    # LSTM Model
    (model, tokenizer) = load_model_tokenizer_lstm("../lstm_classifier.h5", "../lstm_tokenizer.pkl")

    pred_lstm = fact_validator_lstm(fact, model, tokenizer, 300)

    # Ensembling Result
    pred_lstm = float(pred_lstm)
    pred_pac = float(pred_pac[0])

    if(pred_pac == 1.0):
        pred_pac = 0.75
    else :
        pred_pac = 0.25

    ensembled_result = (pred_lstm + pred_pac)/2

    return {"trust_score": str(ensembled_result)}