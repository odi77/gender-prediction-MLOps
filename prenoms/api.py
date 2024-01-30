import flask
from flask import Flask, request
import joblib
import pandas as pd 
import numpy as np
from flask import render_template



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
    
    return vector.reshape(1, -1)



@app.route('/')
def hello():
    return f"Welcome to the gender predictor app"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # If the form is submitted, get the name from the form
        name = request.form['name']
    else:
        # If it's a GET request, show the form to the user
        return render_template('predict.html')

    if name is None or name.strip() == "":
        return flask.jsonify({"error": "Name cannot be empty"}), 400

    results = model.predict(encode_prenom(name))
    prediction = results[0]
    gender = "FEMALE" if prediction == FEMALE else "MALE"

    return flask.jsonify({"gender": gender, "name": name})


if __name__ == '__main__':
    app.run()