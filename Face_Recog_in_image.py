import cv2 as cv
img = cv.imread('harshit.jpeg') # in my case path of image is 'harshit.jpeg' in the same folder where code resides

# for Local Image
resize = cv.resize(img,(500,500),interpolation=cv.INTER_CUBIC)
cv.imshow('result_harshit',resize)
gray = cv.cvtColor(resize,cv.COLOR_BGR2GRAY)
hc = cv.CascadeClassifier('haar_cascade.xml')
faces = hc.detectMultiScale(gray,scaleFactor = 1.1,minNeighbors=3)
print(len(faces))
for (x,y,w,h) in faces:
    cv.rectangle(resize,(x,y),(x+w,y+h),(0,255,0),thickness = 2)
cv.imshow('result_harshit',resize)
cv.waitKey(0)
