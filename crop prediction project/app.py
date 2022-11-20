from flask import Flask,render_template,request
from sklearn import externals
import joblib
import numpy as np

app = Flask(__name__)  # initialization of flask app using python

model = joblib.load('crop.pkl')   # deserialization of model

@app.route('/',methods=['GET','POST']) # @ is a decorator which is used to map url with python function
def home():
    return render_template('pred.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    f1 = float(request.form['f1'])
    f2 = float(request.form['f2'])
    f3 = float(request.form['f3'])   
    f4 = float(request.form['f4'])
    
    l = [f1,f2,f3,f4]
    arr = np.array(l).reshape(1,4)
    predict = model.predict(arr)
    crop = {'rice':0, 'wheat':1, 'Mung Bean':2, 'Tea':3, 'millet':4, 'maize':5, 'Lentil':6,
       'Jute':7, 'Coffee':8, 'Cotton':9, 'Ground Nut':10, 'Peas':11, 'Rubber':12,
       'Sugarcane':13, 'Tobacco':14, 'Kidney Beans':15, 'Moth Beans':16, 'Coconut':17,
       'Black gram':18, 'Adzuki Beans':19, 'Pigeon Peas':20, 'Chickpea':21, 'banana':22,
       'grapes':23, 'apple':24, 'mango':25, 'muskmelon':26, 'orange':27, 'papaya':28,
       'pomegranate':29, 'watermelon':30}
    
    for key,value in crop.items():
        if value == predict:
            x = key
    
    return render_template('pred.html',prediction=x)
    

if __name__ == "__main__":
    app.run(debug=True)