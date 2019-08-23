import cv2
import sys
import random
import os
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    cv2.rectangle(frame, (400, 100), (900, 550), (255,0,0), 2)
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        var1=frame.copy()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        print(x,y,x+w,y+h)
        if(x<800 and x>400 and y<300 and y>100 and (x+w)<900 and (x+w)>400 and (y+h)<560 and (y+h) and len(faces)==1):
            cv2.putText(frame,"Perfect", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #if the face coems inside and teh dimensionsof the video is more than half the size of window
            # the frame will be captured.
            cv2.imwrite(os.path.join('{path to directory of fake images}',"fake%d.jpg" % random.randint(40000,90000)), var1) 
    label = "{}".format(len(faces))
    cv2.imshow('Video', frame)
    # to show frames consecutively
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break     #on pressing the 'q' button the frame capturing will end.

video_capture.release()
cv2.destroyAllWindows()