from flask import Flask,render_template,request
app = Flask(__name__)
import pickle
import numpy as np
model=pickle.load(open('model.pkl','rb'))  
@app.route('/main')
def index():
    return render_template("main.html")
@app.route('/register')
def register():
    return render_template("register.html")
@app.route('/login')
def login():
    return render_template("login.html")
@app.route('/')
def heart():
    return render_template("heart.html")
@app.route('/predict_heart', methods=['POST'])
def predict():
    age = int(request.form.get('age'))
    sex = int(request.form.get('sex'))
    cp = int(request.form.get('cp'))
    trestbps = int(request.form.get('trestbps'))
    chol = int(request.form.get('chol'))
    fbs = int(request.form.get('fbs'))
    restecg = int(request.form.get('restecg'))
    thalach = int(request.form.get('thalach'))
    exang = int(request.form.get('exang'))
    oldpeak = float(request.form.get('oldpeak'))
    slope = int(request.form.get('slope'))
    ca = int(request.form.get('ca'))
    thal = int(request.form.get('thal'))

    result = model.predict(np.asarray([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]).reshape(1,-1))

    if result[0] == 1: 
        prediction = "The patient seems to have heart disease:("
    else:
        prediction = "The patient seems to be Normal:)"

    return render_template('heart_result.html', result=prediction)

if __name__ == "__main__":
    app.run(debug=True)