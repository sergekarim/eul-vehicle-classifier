import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras import regularizers
import warnings

from config import RESULT_PATH, DATA_DIR, EPOCHS, BATCH_SIZE, PREDICTED_DIR

warnings.filterwarnings("ignore")
sns.set_style('darkgrid')

# Define data paths
os.makedirs(RESULT_PATH, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

# File paths for saving results
CHECKPOINT_PATH = os.path.join(RESULT_PATH, "best_model.keras")
LOSS_IMAGE_PATH = os.path.join(RESULT_PATH, 'validation_loss.png')
ACC_IMAGE_PATH = os.path.join(RESULT_PATH, 'validation_accuracy.png')
CONFUSION_IMAGE_PATH = os.path.join(RESULT_PATH, 'confusion_matrix.png')


# Generate file paths and labels
def generate_data_paths(data_dir):
    filepaths, labels = [], []
    for fold in os.listdir(data_dir):
        if fold.startswith('.'):  # Skip hidden files/folders
            continue
        foldpath = os.path.join(data_dir, fold)
        if os.path.isdir(foldpath):
            for file in os.listdir(foldpath):
                if file.startswith('.'):  # Skip hidden files
                    continue
                filepaths.append(os.path.join(foldpath, file))
                labels.append(fold)
    return filepaths, labels

# Create dataframe
def create_dataframe(data_dir):
    filepaths, labels = generate_data_paths(data_dir)
    return pd.DataFrame({'filepaths': filepaths, 'labels': labels})

# Dataset information
def dataset_info(df, name='Dataset'):
    print(f"{name} has {df.shape[0]} images and {df['labels'].nunique()} classes.")
    print(df['labels'].value_counts())


# Model creation function
def create_model(input_shape, num_classes):
    base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape, pooling='max')
    # base_model = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=input_shape, pooling='max')
    base_model.trainable = False

    model = Sequential([
        base_model,
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        Dense(128, kernel_regularizer=regularizers.l2(0.016),
              activity_regularizer=regularizers.l1(0.006),
              bias_regularizer=regularizers.l1(0.006),
              activation='relu'),
        Dropout(rate=0.45, seed=123),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    # Print the model summary
    model.summary()
    return model

# Function for plotting training history
def plot_history(history):
    epochs = range(1, len(history.history['accuracy']) + 1)
    plt.figure(figsize=(12, 5))

    # Plot for loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(LOSS_IMAGE_PATH)

    # Plot for accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['accuracy'], label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(ACC_IMAGE_PATH)

# This function which will be used in image data generator for data augmentation, it just take the image and return it again.
def scalar(img):
    return img

# Function for training the model
def train_model():
    df = create_dataframe(DATA_DIR)
    dataset_info(df, 'Dataset')

    train_df, temp_df = train_test_split(df, train_size=0.8, random_state=123, shuffle=True)
    valid_df, test_df = train_test_split(temp_df, train_size=0.6, random_state=123, shuffle=True)

    dataset_info(train_df, 'Training Set')
    dataset_info(valid_df, 'Validation Set')
    dataset_info(test_df, 'Testing Set')

    img_size = (224, 224)
    batch_size = BATCH_SIZE
    channels = 3
    img_shape = (img_size[0], img_size[1], channels)

    # Model accuracy
    ts_length = len(test_df)
    test_batch_size = max(
        sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length % n == 0 and ts_length / n <= 80]))


    datagen = ImageDataGenerator(
        preprocessing_function=scalar,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True
    )

    train_gen = datagen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels',
                                             target_size=img_size, class_mode='categorical',
                                             color_mode='rgb', shuffle=True, batch_size=batch_size)

    valid_gen = datagen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels',
                                             target_size=img_size, class_mode='categorical',
                                             color_mode='rgb', shuffle=True, batch_size=batch_size)

    test_gen = datagen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels',
                                            target_size=img_size, class_mode='categorical',
                                            color_mode='rgb', shuffle=False, batch_size=test_batch_size)

    model = create_model(img_shape, len(train_gen.class_indices))

    # Callbacks
    model_checkpoint = ModelCheckpoint(CHECKPOINT_PATH, monitor='val_loss', save_best_only=True)

    # Train the model
    history = model.fit(x=train_gen,
                        epochs=EPOCHS,
                        validation_data=valid_gen,
                        validation_steps= None,
                        shuffle= False,
                        batch_size= batch_size,
                        callbacks=[model_checkpoint],
                        verbose=1)

    plot_history(history)

    # Evaluate the model
    train_score = model.evaluate(train_gen)
    valid_score = model.evaluate(valid_gen)
    test_score = model.evaluate(test_gen)

    print(f"Train Accuracy: {train_score[1]}, Validation Accuracy: {valid_score[1]}, Test Accuracy: {test_score[1]}")

    # Confusion matrix and classification report
    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)
    cm = confusion_matrix(test_gen.classes, y_pred)
    print(classification_report(test_gen.classes, y_pred, target_names=list(train_gen.class_indices.keys())))

    # Save confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(train_gen.class_indices.keys()),
                yticklabels=list(train_gen.class_indices.keys()))
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(CONFUSION_IMAGE_PATH)

if __name__ == "__main__":
    train_model()