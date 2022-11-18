# About
This Github repo explores the methodology involved in creating a real-time American sign language (ASL) letters classifier. 

To be able to successfully classify images into letters we propose an Internet-of-things (IoT) system which takes in images in real time and passes these images to a server which then performs the necessary computing by way of machine learning and returns the classified letter. Our final product is able to successfully classify letters in real time at up to 30 frames per second. This proof of concept paves the way for more complex classification involving ASL, such as using words instead of letters to improve the speed of communication.

# How to use our system

## Jupyter Notebooks (Clients)

## machine_learning

This folder contains the code for the entire Machine learning pipeline. It contains code for extracting frames from videos, converting them into file to label mapping for the `ImageDataset`. It also contains the training code for the various models we tried. For this particular project we are using the `PyTorch` framework. [Do note that you may have to change the directory names within the files and download the dataset for the code to work properly]

## WeMos_Client

## FONZ5R7IOSPBB9K.ino

## flask_server
This folder contains the code to run the Flask server with REST API endpoints to communicate with our MongoDB database. To set up the flask server, go into the `flask_server` folder, and run `python3 server.py` which runs the server on `localhost:8000`.

`test.py` gives an example on how you can make the requests to communicate with the server.
