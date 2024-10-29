from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('diabetes_model.pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        pregnancies = int(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        bloodpressure = float(request.form['bloodpressure'])
        skinthickness = float(request.form['skinthickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])

        # Prepare input for the model
        input_data = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]])
        
        # Make prediction
        prediction = model.predict(input_data)

        # Generate prediction text
        if prediction[0] == 1:
            prediction_text = "Diabetes Detected"
        else:
            prediction_text = "No Diabetes Detected"

        return render_template('index.html', prediction_text=prediction_text)

    except Exception as e:
        return render_template('index.html', prediction_text="Error in input data: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)
