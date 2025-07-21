from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the trained machine learning model
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    print("Error: 'model.pkl' not found. Please run the notebook to train and save the model.")
    model = None

@app.route('/')
def home():
    """Renders the home page with the input form."""
    # Change 'index.html' to 'predict_page.html'
    return render_template('predict_page.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the form submission and makes a prediction."""
    if model is None:
        return "Model not loaded. Please check the server logs."

    if request.method == 'POST':
        try:
            # Get form data and convert to the correct types
            age = int(request.form['age'])
            gender = int(request.form['gender'])
            total_bilirubin = float(request.form['total_bilirubin'])
            direct_bilirubin = float(request.form['direct_bilirubin'])
            alkaline_phosphotase = int(request.form['alkaline_phosphotase'])
            alamine_aminotransferase = int(request.form['alamine_aminotransferase'])
            aspartate_aminotransferase = int(request.form['aspartate_aminotransferase'])
            total_protiens = float(request.form['total_protiens'])
            albumin = float(request.form['albumin'])
            albumin_and_globulin_ratio = float(request.form['albumin_and_globulin_ratio'])

            features = np.array([[
                age, gender, total_bilirubin, direct_bilirubin,
                alkaline_phosphotase, alamine_aminotransferase,
                aspartate_aminotransferase, total_protiens, albumin,
                albumin_and_globulin_ratio
            ]])
            
            prediction = model.predict(features)
            prediction_proba = model.predict_proba(features)[0][1]

            if prediction[0] == 1:
                result_text = "The model predicts that the patient is LIKELY to have Liver Cirrhosis."
                confidence = f"Confidence: {prediction_proba*100:.2f}%"
            else:
                result_text = "The model predicts that the patient is UNLIKELY to have Liver Cirrhosis."
                confidence = f"Confidence: {(1-prediction_proba)*100:.2f}%"
                
            # Change 'result.html' to 'result_page.html'
            return render_template('result_page.html', result=result_text, confidence=confidence)

        except Exception as e:
            # Change 'result.html' to 'result_page.html'
            return render_template('result_page.html', result=f"An error occurred: {e}", confidence="")

    # Change 'index.html' to 'predict_page.html'
    return render_template('predict_page.html')

if __name__ == '__main__':
    app.run(debug=True)