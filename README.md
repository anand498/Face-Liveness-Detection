# Face Liveness Detection using Depth Map Prediction

## About the Project
This is an application of a combination of Convolutional Neural Networks and Computer Vision to detect
between actual faces and fake faces in realtime environment. The image frame captured from webcam is passed over a pre-trained model. This model is trained on the depth map of images in the dataset. The depth map generation have been developed from a different CNN model.

Project under continous development.

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
This file contains the CNN architecture of the prediction model and teh various parameter tuning related processes conducted while training the model.

[eye_aspect_ratio.py](https://github.com/anand498/Face-Liveness-Detection/blob/master/eye_aspect_ratio.py):
This is an *optional script* if one needs to add multiple checks like blinking of the eye for the liveness of the video feed.


Will keep updating the repo.
Stay Tuned.
If you use the project please leave a star :)