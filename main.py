from app.controller import app  # Import the Flask app from the controller
from app.model import train_model
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_model()
    else:
        app.run(debug=True, port=5005)