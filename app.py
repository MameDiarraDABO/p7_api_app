from flask import Flask,jsonify, request
import urllib
from urllib.request import urlopen
import requests
import json
import pandas as pd
import numpy as np
import pickle
import sklearn



my_app = Flask(__name__)

# Load Dataframe
dataframe = pd.read_csv("TEST_FINAL_SCALEE.csv")
dataframe = dataframe.iloc[0:1000, :]
data = dataframe.copy()
data = data.drop(['SK_ID_CURR'], axis=1)

#load model
model = pickle.load(open('Credit_model_reg.pkl','rb'))
#liste des identifiants
Id_client = dataframe['SK_ID_CURR'].values

@my_app.route('/api/client/<id_client>', methods=["GET"])
def client(id_client):
    print("id_client:<"+ id_client+">")

    dico = { }
    dico["hello"] = "hello " + str(id_client)

    return jsonify(dico)


@my_app.route('/prediction/client/<id_client>', methods=["GET"])
def predict(id_client):
    id_client = int(id_client)
    Y = dataframe[dataframe['SK_ID_CURR'] == id_client]
    Y = Y.drop(['SK_ID_CURR'], axis=1)
    num = np.array(Y)
    pr = model.predict_proba(num)[:, 1]

    if pr > 0.5:
        prediction = "Rejet de la demande de credit: {:.2f}"


    else:
        prediction = "Acceptation de la demande de credit: {:.2f}"
    resultat = {'Reponse prediction':prediction.format((pr[0] * 100).round())}

    return jsonify(resultat)


if __name__ == "__main__":
    my_app.run(debug=True)