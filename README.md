# Face Liveness Detection using Depth Map Prediction

## About the Project

This is an application of a combination of Convolutional Neural Networks and Computer Vision to detect
between actual faces and fake faces in realtime environment. The image frame captured from webcam is passed over a pre-trained model. This model is trained on the depth map of images in the dataset. The depth map generation have been developed from a different CNN model.



## Requirements

* Python3
* Tensorflow
* dlib
* Keras
* numpy
* sklearn
* Imutils
* OpenCV 


## File Description

[main.py](https://github.com/anand498/Face-Liveness-Detection/blob/master/main.py):
This file is the main script that would call the predictperson function present in the utilr function

[training.py](https://github.com/anand498/Face-Liveness-Detection/blob/master/livenessdetect/training.py):
Along with the architecture script, this file includes various parameter tuning steps of the model.

[model.py](https://github.com/anand498/Face-Liveness-Detection/blob/master/livenessdetect/model.py) :
Has the main CNN architecture for training the dataset

## The Convolutional Neural Network

The network consists of **3** hidden conlvolutional layers with **relu** as the activation function. Finally it has **1** fully connected layer.

The network is trained with **10** epochs size with batch size **32**

The ratio of training to testing bifuracation is **75:25**


### How to use application in real time.


```
git clone https://github.com/anand498/Face-Liveness-Detection.git
pip install -r requirements.txt
python main.py
```
And you're good to go!

Don't forget to  :star:    the repo if I made your life easier with this. :wink:



