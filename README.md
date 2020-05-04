# Face Liveness Detection using Depth Map Prediction

## About the Project

This is an application of a combination of Convolutional Neural Networks and Computer Vision to detect
between actual faces and fake faces in realtime environment. The image frame captured from webcam is passed over a pre-trained model. This model is trained on the depth map of images in the dataset. The depth map generation have been developed from a different CNN model.



## Requirements

* Python3
* Tensorflow
* dlib
* numpy
* Imutils
* OpenCV (cv) for python3


## File Description

[facepredictor.py](https://github.com/anand498/Face-Liveness-Detection/blob/master/facepredictor.py):
This file is the main script that would capture the faces and process them before loading it into the model for prediction. The user needs to ensure that the face should be within the designated frame.

[training.py](https://github.com/anand498/Face-Liveness-Detection/blob/master/training.py):
Along with the architecture script, this file includes various parameter tuning steps of the model.

[eye_aspect_ratio.py](https://github.com/anand498/Face-Liveness-Detection/blob/master/eye_aspect_ratio.py):
This is an *optional script* if one needs to add multiple checks like blinking of the eye for the liveness of the video feed.

[architecture.py](https://github.com/anand498/Face-Liveness-Detection/blob/master/architecture.py) :
Has the main CNN architecture for training the dataset

## The Concolutional Neural Network

The network consists of **3** hidden conlvolutional layers with **relu** as the activation function. Finally it has **1** fully connected layer.

The network is trained with **10** epochs size with batch size **32**

The ratio of training to testing bifuracation is **75:25**


## How to use it in real time.

Clone the complete directory.
[facepredictor.py](https://github.com/anand498/Face-Liveness-Detection/blob/master/facepredictor.py):
Run this file after making the changes in the path of the pre-trained model.

## Future Work
 The whole project is based on the dataset that uses an image dataset that uses faces and the rest of environment behing it. That brings a lot of factors and elements into consideration for prediction.
 In the next version of this directory, I will make a model that is only trained on the **face features** of the test image by cropping the rest of the enivronment.
Will keep updating the repo.
Stay Tuned.
