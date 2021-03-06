
# Import OpenCV2 for image processing
import cv2
import os
import time

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Start capturing video 
video_cam = cv2.VideoCapture(-1)

# check if capture open is enabled
print(video_cam.isOpened())

time.sleep(5)

# Detect object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier('/home/pi/opencv-3.3.0/data/haarcascades/haarcascade_frontalface_default.xml')

# For each person, one face id
face_iden = 1

# Initialize sample face image
count = 0

assure_path_exists("dataset/")

# Start looping
while(True):

    # Capture video frame
    _, img_frme = video_cam.read()

    # Convert frame to grayscale
    gray_img = cv2.cvtColor(img_frme, cv2.COLOR_BGR2GRAY)
    
    # Detect frames of different sizes, list of faces rectangles
    faces = face_detector.detectMultiScale(gray_img, 1.3, 5)
    print(faces)
    # Loops for each faces
    for (x,y,w,h) in faces:

        # Crop the image frame into rectangle
        cv2.rectangle(img_frme, (x,y), (x+w,y+h), (255,0,0), 2)
        
        # Increment sample face image
        count += 1
        print(count)
        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_iden) + '.' + str(count) + ".jpg", gray_img[y:y+h,x:x+w])

        # Display the video frame, with bounded rectangle on the person's face
        cv2.imshow('frame', img_frme)

    # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

    # If image taken reach 50, stop taking video
    elif count>50:
        break

# Stop video
video_cam.release()

# Close all started windows
cv2.destroyAllWindows()
