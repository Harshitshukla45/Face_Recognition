import cv2 as cv
img = cv.VideoCapture(1)  # 0 - internal (laptop) camera , 1 - Exteranal(webcam) camera and for local video path should be provided instead of 0 1 2...
hc = cv.CascadeClassifier('haar_cascade.xml')

# For video capturing from internal/external camera
while True:
    isTrue, frame = img.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = hc.detectMultiScale(gray,scaleFactor = 1.1,minNeighbors=3)
    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness = 2)
    cv.imshow('Harshit',frame)
    if(cv.waitKey(20) & 0xFF == ord('q')):
        break
