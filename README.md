# Face-Recognization-Application
An end-to-end Python application for collecting face data, training a recognition model, and identifying faces in real time using Numpy, OpenCV.

# Key Features

Live Data Collection: The app can capture and save a custom number of face samples from a live camera feed to build a robust dataset for training.
Model Training: It uses the Local Binary Patterns Histograms (LBPH) algorithm to train a facial recognition model, saving the trained data as trained_data.xml for quick loading.
Real-time Recognition: The application performs real-time face detection and classifies individuals as either "Known" or "Unknown" based on a confidence score.
Flexible Camera Input: The app supports both a standard webcam and a mobile phone camera using the DroidCam application, providing an accessible and flexible video source.
User-Friendly Interface: A simple command-line interface allows users to easily choose between collecting data, training the model, and testing the recognition system.

# Technologies Used

Python: The core programming language.
OpenCV: For all computer vision tasks, including face detection (cv2.CascadeClassifier), image processing, and training the recognizer (cv2.face.LBPHFaceRecognizer).
NumPy: Used for efficient numerical operations and data manipulation on image arrays.
DroidCam: An application used to connect a mobile phone as a camera source, providing a flexible alternative to a standard webcam.

# Getting Started
Prerequisites
Before running the application, you need to have Python and the required libraries installed.

Bash
# pip install opencv-python numpy

# Cloning the Repository
First, clone this repository to your local machine:

Bash
# git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Replace your-username/your-repo-name with your actual GitHub repository information.

# How to Use the Application
To start the application, simply run the main.py file from your terminal:

Bash
# python main.py

# You will be presented with a command-line menu with the following options:

Press 1 to Collect Sample Faces: This will open your camera and begin capturing 100 images of your face.

Press 2 to Train the Model: This action uses the images in the faces/ directory to train the recognition model.

Press 3 to Test the Model: This will open a new camera window and perform real-time face recognition.

Press 4 to Exit the App.

# Project Structure

main.py: The main application script containing the logic for all features.
haarcascade_frontalface_default.xml: The pre-trained XML file for face detection.
trained_data.xml: (Generated after training) The trained face recognition model.
faces/: (Generated after collecting faces) A directory where the sample images are stored.
