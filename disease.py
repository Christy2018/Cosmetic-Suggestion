import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import PIL
import pathlib
import cv2
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from tensorflow import keras 
import tf_keras as tfk

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
img_size = (224,224)
# Load the array back
x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")
mobileNet = "https://www.kaggle.com/models/google/mobilenet-v2/TensorFlow2/tf2-preview-feature-vector/4"
mNet = hub.KerasLayer(mobileNet,input_shape=img_size+(3,),trainable=False)
model  = tfk.Sequential([
    mNet,
    tfk.layers.Dense(170,'relu'),
    tfk.layers.Dense(8,'softmax')
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=["accuracy"])
model.fit(x_train,y_train,epochs=5)
model.evaluate(x_test,y_test)
yp= model.predict(x_test)

labels = {
    "BA-cellulitis": 0,
    "BA-impetigo": 1,
    "FU-athlete-foot": 2,
    "FU-nail-fungus": 3,
    "FU-ringworm": 4,
    "PA-cutaneous-larva-migrans": 5,
    "VI-chickenpox": 6,
    "VI-shingles": 7    
}

# Invert the dictionary to map from index to label name.
inv_labels = {v: k for k, v in labels.items()}

# Define the image size that the model expects.
img_size = (224, 224)

# Path to the new image you want to classify.
img_path = r"test_image.jpeg"

# Load the image using OpenCV.
img = cv2.imread(img_path)

# Check if image could be loaded.
if img is None:
    raise ValueError("Image not found or unable to load. Check the file path:", img_path)

# Resize the image to the expected input size.
img_resized = cv2.resize(img, img_size)

# Convert the image from BGR to RGB if necessary (OpenCV loads in BGR by default).
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

# Normalize the image by scaling pixel values to [0, 1].
img_normalized = img_rgb / 255.0

# Expand dimensions to create a batch of one image.
img_input = np.expand_dims(img_normalized, axis=0)  # Shape becomes (1, 224, 224, 3)

# Predict the class probabilities for the input image.
predictions = model.predict(img_input)

# Get the index of the highest probability.
predicted_index = np.argmax(predictions, axis=1)[0]

# Map the predicted index back to the class name using the inverse dictionary.
predicted_label = inv_labels[predicted_index]

print("Predicted Class:", predicted_label)
