import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import plot

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

warnings.filterwarnings("ignore")
sns.set_style('darkgrid')

# Define data paths
data_dir = '/Users/sergekarim/Documents/UNICAF/MSC/Artificial Intelligence/code/Dataset'
ds_name = 'Vehicles'
result_path = f"/Users/sergekarim/Documents/UNICAF/MSC/Artificial Intelligence/code/working/run/"

os.makedirs(result_path, exist_ok=True)
checkpoint_path = os.path.join(result_path, "best_model.keras")
loss_image_path = os.path.join(result_path, 'validation_loss.png')
acc_image_path = os.path.join(result_path, 'validation_accuracy.png')
confusion_image_path = os.path.join(result_path, 'confusion_matrix.png')


# Generate file paths and labels
def generate_data_paths(data_dir):
    filepaths, labels = [], []
    for fold in os.listdir(data_dir):
        foldpath = os.path.join(data_dir, fold)
        if os.path.isdir(foldpath):
            for file in os.listdir(foldpath):
                filepaths.append(os.path.join(foldpath, file))
                labels.append(fold)
    return filepaths, labels


filepaths, labels = generate_data_paths(data_dir)

# Create dataframe
df = pd.DataFrame({'filepaths': filepaths, 'labels': labels})


# Display dataset info
def dataset_info(df, name='Dataset'):
    print(f"{name} has {df.shape[0]} images and {df['labels'].nunique()} classes.")
    print(df['labels'].value_counts())


dataset_info(df, ds_name)

# Train, validation, test split
train_df, temp_df = train_test_split(df, train_size=0.8, random_state=123, shuffle=True)
valid_df, test_df = train_test_split(temp_df, train_size=0.6, random_state=123, shuffle= True)

dataset_info(train_df, 'Training Set')
dataset_info(valid_df, 'Validation Set')
dataset_info(test_df, 'Testing Set')

# Image Data Generators
img_size = (224, 224)
batch_size = 16
channels = 3
img_shape = (img_size[0], img_size[1], channels)

# Model accuracy
ts_length = len(test_df)
test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length%n == 0 and ts_length/n <= 80]))
test_steps = ts_length // test_batch_size

# This function which will be used in image data generator for data augmentation, it just take the image and return it again.
def scalar(img):
    return img

datagen = ImageDataGenerator(preprocessing_function=scalar,
                             rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             zoom_range=0.3,
                             horizontal_flip=True,
                             vertical_flip=True)

train_gen = datagen.flow_from_dataframe(train_df,
                                       x_col= 'filepaths',
                                       y_col= 'labels',
                                       target_size= img_size,
                                       class_mode= 'categorical',
                                       color_mode= 'rgb',
                                       shuffle= True,
                                       batch_size= batch_size)
valid_gen = datagen.flow_from_dataframe(valid_df,
                                       x_col= 'filepaths',
                                       y_col= 'labels',
                                       target_size= img_size,
                                       class_mode= 'categorical',
                                       color_mode= 'rgb',
                                       shuffle= True,
                                       batch_size= batch_size)

test_gen = datagen.flow_from_dataframe(test_df,
                                      x_col= 'filepaths',
                                      y_col= 'labels',
                                      target_size= img_size,
                                      class_mode= 'categorical',
                                      color_mode= 'rgb',
                                      shuffle= False,
                                      batch_size= test_batch_size)

# Define the model (EfficientNetB0)
base_model = tf.keras.applications.EfficientNetB7(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')
base_model.trainable = False

model = Sequential([
    base_model,
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
    Dense(128,
          kernel_regularizer=regularizers.l2(0.016),
          activity_regularizer=regularizers.l1(0.006),
          bias_regularizer=regularizers.l1(0.006),
          activation='relu'),
    Dropout(rate=0.45, seed=123),
    Dense(len(train_gen.class_indices), activation='softmax')
])

model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True,mode='max',)

def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

lr_scheduler = LearningRateScheduler(step_decay)

# Train the model
batch_size = 16   # set batch size for training
epochs = 15   # number of all epochs in training
history = model.fit(x=train_gen,
                    epochs=epochs,
                    verbose= 1,
                    validation_data=valid_gen,
                    callbacks=[model_checkpoint],
                    validation_steps= None,
                    shuffle= False,
                    batch_size= batch_size
                    )


# Plot accuracy and loss
def plot_history(history, loss_image_path, acc_image_path):
    epochs = range(1, len(history.history['accuracy']) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.savefig(loss_image_path)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['accuracy'], label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.savefig(acc_image_path)
    plt.show()


plot_history(history, loss_image_path, acc_image_path)

# Evaluate model
train_score = model.evaluate(train_gen)
valid_score = model.evaluate(valid_gen)
test_score = model.evaluate(test_gen)

print(f"Train Accuracy: {train_score[1]}, Test Accuracy: {test_score[1]}")

# Confusion matrix and classification report
preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)
cm = confusion_matrix(test_gen.classes, y_pred)
print(classification_report(test_gen.classes, y_pred, target_names=list(train_gen.class_indices.keys())))

# Save confusion matrix
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(train_gen.class_indices.keys()),
            yticklabels=list(train_gen.class_indices.keys()))
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(confusion_image_path)
plt.show()
