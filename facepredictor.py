from keras.preprocessing.image import img_to_array
from keras.models import load_model
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import cv2
import os
import sys
import dlib

def predictperson():
	video_capture = cv2.VideoCapture(0)
	countreal=0
	countfake=0
	countframe=0
	while(True):
		countframe+=1
		if cv2.waitKey(1) & 0xFF == ord('q'):
					break 
		if	cv2.waitKey(1) & 0xFF == ord('r'):
				countreal=0
				countfake=0
		
		ret,frame = video_capture.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),)
		cv2.rectangle(frame, (400, 100), (900, 550), (255,0,0), 2)
		cv2.putText(frame,"Please keep your head inside the blue box and have only one face in the frame", (10, 700),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		
		faces_inside_box = 0
		for (x, y, w, h) in faces:
			if x<800 and x>400 and y<300 and y>100 and (x+w)<900 and (x+w)>400 and (y+h)<560 and (y+h)>100:
				faces_inside_box+=1
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

		if faces_inside_box > 1 :
			cv2.putText(frame,"Multiple Faces detected!", (600, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		if faces_inside_box == 1 :
			(x, y, w, h)
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

			if(w*h > (500*450)/4 ) and x<800 and x>400 and y<300 and y>100 and (x+w)<900 and (x+w)>400 and (y+h)<560 and (y+h)>100:
				#cv2.putText(frame,"Perfect", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				image = cv2.resize(frame, (128, 128))
				image = image.astype("float") / 255.0
				image = img_to_array(image)
				image = np.expand_dims(image, axis=0)
				(real, fake) = model.predict(image)[0]
				if fake > real:
					label = "fake"
					countfake+=1
				else:
					label= "real"
					countreal+=1
				proba_real=real
				proba_fake=fake
				proba = fake if fake > real else real
				label = "{}".format(label)
				proba_real="{}".format(proba_real)
				proba_fake="{}".format(proba_fake)
				countf="{}".format(countfake)
				countr="{}".format(countreal)
				#if(countframe%10==0):
					#if abs(countfake-countreal)>10:
				cv2.putText(frame,label, (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
					#else:
					#	cv2.putText(frame,"Error. Wrong Identification", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				#cv2.putText(frame,"Real Count", (10, 180),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				#cv2.putText(frame,countr, (150, 180),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				#cv2.putText(frame,"Fake Count", (10, 220),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				#cv2.putText(frame,countf, (150, 220),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)	
			else:
				cv2.putText(frame,"Please come closer to the camera", (10, 390),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			
				
		cv2.imshow("Frame",frame)
			
	

if __name__ == '__main__':

	model = load_model('anandfinal.hdf5')
	cascPath = "haarcascade_frontalface_default.xml"
	faceCascade = cv2.CascadeClassifier(cascPath)
	
	predictperson()
	