import os
from calendar import EPOCH

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')  # Update this path
RESULT_PATH = os.path.join(BASE_DIR, 'results')  # Update this path
MODEL_PATH = os.path.join(RESULT_PATH, 'best_model.keras')
PREDICTED_DIR = os.path.join(RESULT_PATH, 'predicted')

# Callbacks settings
CHECKPOINT_PATH = os.path.join(RESULT_PATH, "best_model.keras")
EPOCHS = 30
BATCH_SIZE= 16
CLASSES = ['Bus', 'Car','Motorcycle', 'SmallTruck', 'Truck']
