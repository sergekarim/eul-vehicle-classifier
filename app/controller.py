from flask import Flask, request, jsonify
from PIL import Image
from app.model_utils import predict_image, preprocess_image, load_model, save_predicted_image
from config import MODEL_PATH

app = Flask(__name__)

# Load the model once
model = load_model(MODEL_PATH)


@app.route('/upload', methods=['POST'])
def upload_and_predict():
    # Check if an image file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    # Check if the file is a valid image
    if file and file.filename.endswith(('png', 'jpg', 'jpeg')):
        # Open the image
        image = Image.open(file)
        image_array = preprocess_image(image)
        # Make a prediction
        predictionResult =  predict_image(image_array,model)

        # Save result image
        save_predicted_image(file,predictionResult[0],predictionResult[1])

        return jsonify({'prediction': predictionResult[0],'confidence': predictionResult[1],})
    else:
        return jsonify({'error': 'Invalid file type. Only png, jpg, and jpeg are allowed.'}), 400

