from flask import Flask, request, render_template
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')  # Ensure the model.pkl file is in the same directory

# Define crop images for frontend display
crop_images = {
    'rice': 'static/images/rice.jpg',
    'jute': 'static/images/jute.jpg',
    'cotton': 'static/images/cotton.jpg',
    'maize': 'static/images/maize.jpg',
    'lentil': 'static/images/lentil.jpg',
    'mango': 'static/images/mango.jpg',
    'default': 'static/images/default.jpg'  # For crops without images
}

# Define the homepage route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Prepare the features for prediction
        features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])

        # Predict the crop
        crop_prediction = model.predict(features)[0]

        # Get the image for the predicted crop
        crop_image = crop_images.get(crop_prediction, crop_images['default'])

        # Render the result page
        return render_template('result.html', crop=crop_prediction, crop_image=crop_image)
    except Exception as e:
        return render_template('index.html', error="Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)
