
# Import OpenCV2 for image processing
import cv2

# Import numpy for matrices calculations
import numpy as np

import os

#from num2words import num2words
from subprocess import call

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Create Local Binary Patterns Histograms for face recognization
recognizer_patterns = cv2.face.LBPHFaceRecognizer_create()

count=1

assure_path_exists("trainer/")

# Load the trained mode
recognizer_patterns.read('trainer/trainer.yml')

# Load prebuilt model for Frontal Face
cascade = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascade);

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture
caminput = cv2.VideoCapture(0)

# Loop
while True:
    # Read the video frame
    ret, im =caminput.read()

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # Get all face from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.2,5)

    # For each face in faces
    for(x,y,w,h) in faces:

        # Create rectangle around the face
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

        # Recognize the face belongs to which ID
        user, confidence = recognizer_patterns.predict(gray[y:y+h,x:x+w])

        # Check the ID if exist 
        if(user == 1):
            user = "Nithin {0:.2f}%".format(round(100 - confidence, 2))
            if(count==1):
                oo='espeak "Hello Nithin" 2>/dev/null'
                call([oo], shell=True)
                count=count+1                    


        # Put text describe who is in the picture
        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, str(user), (x,y-40), font, 1, (255,255,255), 3)

    # Display the video frame with the bounded rectangle
    cv2.imshow('im',im)
    

    # If 'q' is pressed, close program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Stop the camera
caminput.release()

# Close all windows
cv2.destroyAllWindows()
