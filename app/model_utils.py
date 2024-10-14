import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf
import os
import uuid
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import CLASSES, PREDICTED_DIR

# Load the Keras model (replace with your model path)
def load_model(model_path):
    """Load the trained model from the specified path."""
    return tf.keras.models.load_model(model_path)

# Define the classes for prediction

# Helper function to preprocess image
def preprocess_image(image):
    """Resize and normalize the image."""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Helper function to predict image

def predict_image(img_array,model):
    # Get the prediction probabilities for each class
    prediction = model.predict(img_array)

    # Find the index of the class with the highest predicted probability
    predicted_class_index = np.argmax(prediction)

    # Get the label of the predicted class
    predicted_class_label = CLASSES[predicted_class_index]

    # Get the confidence (probability) of the predicted class
    confidence = prediction[0][predicted_class_index]

    # Return both the predicted class label and the confidence score as a percentage
    return predicted_class_label, confidence * 100



def save_predicted_image(file,predicted_class_label,prediction_confidence):
    image = Image.open(file)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Predicted Vehicle: {predicted_class_label} | Confidence: {prediction_confidence:.2f}%")

    # Generate a random image name
    random_image_name = str(uuid.uuid4()) + '.jpg'
    PREDICTED_IMAGE_PATH = os.path.join(PREDICTED_DIR, random_image_name)

    # Save the figure to the specified path
    plt.savefig(PREDICTED_IMAGE_PATH, bbox_inches='tight', pad_inches=0.15)
    plt.close()  # Close the figure to release memory

