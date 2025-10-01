from contextlib import asynccontextmanager
from fastapi import FastAPI
import asyncio
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as tfk

# Global variables for the model
model = None

# Define the lifespan function
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    
    print("Initializing model during startup...")
    
    # Startup logic: Load and initialize the model
    await asyncio.to_thread(initialize_model)
    
    print("Model initialized successfully.")
    yield  # App will run here
    
    # Shutdown logic: Clean up resources if needed
    print("Shutting down...")
    model = None
    print("Model resources cleaned up.")

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

# Function to train/load the model
def initialize_model():
    global model

    img_size = (224, 224)
    x_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")
    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")

    mobileNet = "https://www.kaggle.com/models/google/mobilenet-v2/TensorFlow2/tf2-preview-feature-vector/4"
    mNet = hub.KerasLayer(mobileNet, input_shape=img_size + (3,), trainable=False)

    model = tfk.Sequential([
        mNet,
        tfk.layers.Dense(170, activation='relu'),
        tfk.layers.Dense(8, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)
    print("Model training and evaluation completed.")

# Endpoint to check if the model is initialized
@app.get("/")
async def check_model():
    if model is not None:
        return {"status": "Model is loaded"}
    else:
        return {"status": "Model is not loaded"}
