from keras.models import Sequential
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers.core import Activation,Dropout,Flatten,Dense
from keras import backend as K

def  MiniVGG(width,height,depth,classes):
		model=Sequential()
		inputS=(height,width,depth)
		chanDim=-1
		if(K.image_data_format()=="channels_first"):
			inputS=(depth,height,width)
			chanDim=-1
		model.add(Conv2D(32,(3,3),padding="same",input_shape=inputS))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(32,(3,3),padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))
		model.add(Conv2D(64,(3,3),padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		model.add(Dense(classes))
		model.add(Activation("softmax"))
		return model


