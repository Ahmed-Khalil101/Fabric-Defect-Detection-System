# Fabric-Defect-Detection-System
The Automated Fabric Defect Detection Tool helps textile manufacturers identify fabric defects (tears, spots, weave inconsistencies) using image processing techniques like edge detection, thresholding, and contour analysis. It automates inspection, improves accuracy, and reduces reliance on manual checks, enhancing quality control efficiency.
Small Description of the Code
This Python script is a complete pipeline for fabric defect classification using a Convolutional Neural Network (CNN). It includes the following key parts:

Imports: Loads essential libraries like TensorFlow, NumPy, OpenCV, and others for image processing, data visualization, and model building.

Synthetic Dataset Generator (create_synthetic_dataset):

If no real dataset is found, it generates artificial images for 4 classes: hole, horizontal, vertical, and normal.

Adds defects like holes or lines to mimic real-world fabric issues.

Data Preprocessing (load_and_preprocess_data):

Reads the image files.

Resizes to 224x224 and normalizes pixel values.

Converts class labels to one-hot encoded format.

CNN Model (create_cnn_model):

A sequential model with 3 convolutional blocks.

Ends with dense layers and softmax activation for multi-class classification.

Model Training & Evaluation (main, plot_training_history, evaluate_model):

Splits data into training and testing sets.

Trains the CNN model with early stopping.

Plots training curves and confusion matrix.

Prints a classification report.

Saves the model to a file
