import flask
from flask import Flask, escape, request
import joblib
import pandas as pd 
import numpy as np


app = Flask(__name__)
model = joblib.load("model.v1.bin")

MALE = 0
FEMALE = 1


# Encoding the input to the model

def encode_prenom(prenom):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    vector_size = len(alphabet)
    
    # Initialiser un vecteur de zéros de la taille de l'alphabet
    vector = np.zeros(vector_size, dtype=int)
    
    # Convertir le prénom en minuscules pour éviter la sensibilité à la casse
    prenom = prenom.lower()
    
    # Remplir le vecteur en fonction des lettres présentes dans le prénom
    for letter in prenom:
        if letter in alphabet:
            index = alphabet.index(letter)
            vector[index] += 1
    
    return pd.Series(vector, index=list(alphabet))

@app.route('/')
def hello():
    return f"Welcome to the gender predictor app"


@app.route('/predict')
def predict():
    name = request.args.get("name")

    results = model.predict(encode_prenom([name]))
    prediction = results[0]
    gender = "FEMALE" if prediction == FEMALE else "MALE"

    return flask.jsonify({"gender": gender, "name": name})