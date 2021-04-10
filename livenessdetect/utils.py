from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import os,cv2


model = load_model("livenessdetect/models/anandfinal.hdf5")
faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def predictperson():
	video_capture = cv2.VideoCapture(0)

	while(True):
		if cv2.waitKey(1) & 0xFF == ord('q'):
					break 
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
			if x<800 and x>400 and y<300 and y>100 and (x+w)<900 and (x+w)>400 and (y+h)<560 and (y+h)>100:
				image = cv2.resize(frame, (128, 128))
				image = image.astype("float") / 255.0
				image = img_to_array(image)
				image = np.expand_dims(image, axis=0)
				(real, fake) = model.predict(image)[0]
				if fake > real:
					label = "fake"
				else:
					label= "real"
				cv2.putText(frame,label, (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

			else:
				cv2.putText(frame,"Please come closer to the camera", (10, 390),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.imshow("Frame",frame)

