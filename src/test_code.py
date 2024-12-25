#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 00:01:48 2024

@author: jonathansetiawan
"""
import cv2
from keras.models import model_from_json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from keras.utils import to_categorical
import os
import glob

# Path to Testing Data
path_image = './data/Testing_set'

# Function to Read and Preprocess Testing Data
def Read_data(path_image):
    array = []
    label = []
    folder = os.listdir(path_image)
    
    for j, out_folder in enumerate(folder):
        image_path = os.path.join(path_image, out_folder)
        image_list = glob.glob(image_path + '/*.jpg')
        
        for i, image in enumerate(image_list):
            img = cv2.imread(image)
            resized_image = cv2.resize(img, (224, 224))
            array.append(resized_image)
            label.append(int(j))  
            # Assign a numeric label based on folder index
    
    # Convert to numpy arrays and normalize
    x_test = np.asarray(array, dtype='float32') / 255.0
    y_test = np.asarray(label, dtype='int')
    yy_test = y_test.copy()  # For calculating accuracy later
    
    # One-hot encoding
    y_test = to_categorical(y_test)
    
    return x_test, y_test, yy_test

# Function to Load the Model and Weights
def reload():
    # Debugging: Print current working directory and list files
    print("Current Working Directory:", os.getcwd())
    print("Files in Directory:", os.listdir())
    
    try:
        # Load the model architecture
        with open('model.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        
        loaded_model = model_from_json(loaded_model_json)
        
        # Load model weights (update with correct file name/path)
        weight_file = "ckpt.weights.h5"  
        # Ensure this file exists in the same directory
        if not os.path.exists(weight_file):
            raise FileNotFoundError(f"Weight file '{weight_file}' not found.")
        
        loaded_model.load_weights(weight_file)
        print("Model and weights loaded successfully!")
        loaded_model.summary()
        
        return loaded_model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

# Function to Evaluate Model on Testing Data
def Score():
    # Load Testing Data
    print("Reading testing data...")
    X_test, Y_test, YY_test = Read_data(path_image)
    print("Testing data loaded successfully!")
    
    # Reload the trained model
    print("Loading the model and weights...")
    loaded_model = reload()
    
    # Predict on Test Data
    print("Performing predictions...")
    Y_pred = loaded_model.predict(X_test).argmax(axis=1)  
    # Get predicted class indices
    
    # Calculate accuracy
    score = accuracy_score(YY_test, Y_pred)
    print("\n=== Model Performance ===")
    print("Accuracy:", score)
    
    # Classification Report
    print("\nClassification Report:\n", classification_report(YY_test, Y_pred))
    

# Run the Evaluation
if __name__ == "__main__":
    Score()
