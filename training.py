
import matplotlib
matplotlib.use("Agg")
from architecture import MiniVGG
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from sklearn.metrics import classification_report,confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

EPOCHS = 10
INIT_LR = 1e-3 #Initial Learning rate
BS = 32 # Bach size to feed

# initialize the data and labels
print("Load images' NPY file")
data = []
labels = []
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(r"{path of the image dataset directory}")))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (128, 128))
	image = img_to_array(image)
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	label = 1 if label == "fake" else 0
	labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
np.save('data.npy',data)
labels = np.array(labels)
np.save('labels.npy',labels)
data=np.load('data.npy')
labels=np.load('labels.npy')
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("Compiling model...")
model = MiniVGG.build(width=128, height=128, depth=3, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS) #Optimise uisng Adam 
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

# train the network
print("Training network")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX),
	epochs=EPOCHS, verbose=1)
label_name=["real","fake"]
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128) 
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1)))

cm = confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1))
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
print(cm)

N = EPOCHS

