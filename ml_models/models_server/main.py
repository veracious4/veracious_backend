from typing import Union

from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import pymongo
import uuid
import pika
import threading

from utils import *


API_DOC_TITLE = "Veracious API"
app = FastAPI( title=API_DOC_TITLE, description="Backend API's of Veracious application")

# Enabling CORS options.
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Connecting to MongoDB
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["veracious_db"]
collection = db["fact_collection"]

# Connecting to RabbitMQ
rabbitmq_connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))

sender_channel = rabbitmq_connection.channel()
sender_channel.queue_declare(queue='fact_validation_req_queue')



#Defining Api end-points
@app.get('/', tags=["General"])
def get_root():
    return {"greetings": "Welcome to Veracious ML model's API"}

@app.get('/validate-fact', tags=["Fact Validation"])
def get_fact_validation(fact: str):
    
    # PAC Model
    (model, vectorizer) = load_model_vectorizer("../passive_aggressive_model.pkl", "../passive_aggressive_vectorizer.pkl")
    
    pred_pac = fact_validator(fact, model, vectorizer)

    # LSTM Model
    (model, tokenizer) = load_model_tokenizer_lstm("../lstm_classifier.h5", "../lstm_tokenizer.pkl")

    pred_lstm = fact_validator_lstm(fact, model, tokenizer, 300)

    # SVM Model
    #(svm_model, svm_vectorizer) = load_model_vectorizer_svm("../saved_models/svm_classifier.pkl", "../saved_models/vectorizer_SVM_NB.pickle")
    #pred_svm = fact_validator_svm(fact, svm_model, svm_vectorizer)

    # Naive Bayes Model
    (nb_model, nb_vectorizer) = load_model_vectorizer_svm("../saved_models/bernoulli_nb_classifier.pkl", "../saved_models/vectorizer_SVM_NB.pickle")
    pred_nb = fact_validator_nb(fact, nb_model, nb_vectorizer)

    # # Bert Model
    # bert_model = load_model_bert("../saved_models/bert/bert_classifier")
    # pred_bert = fact_validator_bert(fact, bert_model)  

    # Ensembling Result
    pred_lstm = float(pred_lstm)
    pred_pac = float(pred_pac[0])
    #pred_svm = float(pred_svm)
    pred_nb = float(pred_nb)
    # pred_bert = float(pred_bert)

    if(pred_pac == 1.0):
        pred_pac = 0.75
    else :
        pred_pac = 0.25
    
    '''if(pred_svm==1.0):
        pred_svm = 0.75
    else:
        pred_svm = 0.25'''

    ensembled_result = (pred_lstm + pred_pac + pred_nb)/3
    # ensembled_result = (pred_lstm + pred_pac + pred_nb + pred_bert)/4

    return {"trust_score": str(ensembled_result)}



@app.get('/validate-fact-async', status_code=status.HTTP_202_ACCEPTED, tags=["Fact Validation"])
def register_fact_validation_request(fact: str):
    '''
    Use this endpoint to register a fact validation request.
    The request will be processed asynchronously and the result will be available at a later time.
    Use the correlation id returned by this endpoint to check the status of the request.
    
    Return
    -----------
        correlation_id : string
            A request id for fact validation. 
    '''
    correlation_id = str(uuid.uuid4())
    sender_channel.basic_publish(exchange='', routing_key='fact_validation_req_queue', body=fact, 
                                 properties=pika.BasicProperties(correlation_id=correlation_id))
    # collection.insert_one({"correlation_id": correlation_id, "status": "pending"})
    return {"correlation_id": correlation_id}



@app.get('/validate-fact-async-status', tags=["Fact Validation"])
def get_fact_validation_status(correlation_id: str, response: Response):
    '''
    Use this endpoint to check the status of the request.
    If the status is pending, then the request is being processed.
    If the status is completed, then the request has been processed and the result is available.
    Keep polling this endpoint until the status is completed.

    Return
    -----------
        status : string
            The status of the request. One of the value from ["error", "pending", "completed"]
        message : string
            A message describing the status of the request.
        result : string
            The result of the request. This field will be present only if the status is completed.
    '''
    result = collection.find_one({"correlation_id": correlation_id})
    if result is None:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {"status": "error", "message": "Invalid correlation id"}
    elif result["status"] == "pending":
        response.status_code = status.HTTP_102_PROCESSING
        return {"status": "pending", "message": "Request is being processed"}
    else:
        return {"status": "completed", "message":"Your result has been successfully processed", "result": result["result"]}



def on_request_message_received(ch, method, properties, body):
    '''
    A callback function that will be called when a message is received on the fact_validation_req_queue.
    This function will process the request and store the results in a MongoDB collection.
    '''
    print(f"Received Request: {properties.correlation_id}")
    fact = body.decode("utf-8")
    response = get_fact_validation(fact)
    print(response)

    # #  Update status in MongoDB
    # collection.update_one({"correlation_id": properties.correlation_id}, 
    #                       {"$set": {"status": "completed", "result": response}})


sender_channel.basic_consume(queue='fact_validation_req_queue', auto_ack=True,
                             on_message_callback=on_request_message_received)

thread1 = threading.Thread(target=sender_channel.start_consuming)
thread1.start()
thread1.join(0)