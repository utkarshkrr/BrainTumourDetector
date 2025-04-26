from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from utils import load_model, preprocess_image, get_prediction
import torch

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB

# Load the model
model, device = load_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Secure the filename
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the file
        file.save(file_path)
        
        # Read file for prediction
        with open(file_path, 'rb') as f:
            img_data = f.read()
        
        # Preprocess the image
        img_tensor = preprocess_image(img_data)
        
        # Get prediction
        prediction, confidence = get_prediction(model, img_tensor, device)
        
        image_url = 'uploads/' + filename
        
        return render_template('result.html', 
                               prediction=prediction,
                               confidence=confidence,
                               image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)