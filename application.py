from flask import Flask,request,jsonify,render_template
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pickle

Model = pickle.load(open('models/ridge.pkl','rb'))
Scaler = pickle.load(open('models/scaler.pkl','rb'))

print("Model:", type(Model))
print("Scaler:", type(Scaler))

application = Flask(__name__)
app = application

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predictfwi",methods=['GET','POST'])
def predictFwi():
    if request.method == 'GET':
        return render_template('form.html')
    if request.method == 'POST':
        temperature = float(request.form.get('Temperature'))
        rh = float(request.form.get('RH'))
        ws = float(request.form.get('Ws'))
        rain = float(request.form.get('Rain'))
        ffmc = float(request.form.get('FFMC'))
        dmc = float(request.form.get('DMC'))
        dc = float(request.form.get('DC'))
        isi = float(request.form.get('ISI'))
        bui = float(request.form.get('BUI'))
        classes = int(request.form.get('Classes'))
        region = int(request.form.get('Region'))

    print("Model:", type(Model))
    print("Scaler:", type(Scaler))

    X_test_standard = Scaler.transform([[temperature,rh,ws,rain,ffmc,dmc,dc,isi,bui,classes,region]])
    y_predicted = Model.predict(X_test_standard)

    return render_template('form.html',prediction = y_predicted[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0")