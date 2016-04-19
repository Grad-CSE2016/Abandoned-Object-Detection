import numpy as np
import imutils
import cv2

def is_human(image):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    image = imutils.resize(image,width = min(400,image.shape[1]))
    orig = image.copy()
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    isPerson = False

    if image.shape[1] < 64:
        image.resize(image.shape[0],65)
    if image.shape[0] < 128:
        image.resize(130,image.shape[1])

    (rects, weights) = hog.detectMultiScale(image,winStride=(8,8), padding = (8,8) , scale = 1.03)
    if type(rects) is not tuple and rects.size > 0:
        isPerson = True
    else:
        rects = detect_face(gray , scaleFactor = 1.1 , minNeighbors = 5 , minSize = (30,30))
        if len(rects) > 0:
            isPerson = True
        else:
            isPerson = False

    return isPerson

def detect_face(image,scaleFactor = 1.1 , minNeighbors = 5 , minSize = (30,30)):
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    rects = faceCascade.detectMultiScale(image,scaleFactor = scaleFactor ,
    minNeighbors = minNeighbors , minSize = minSize , flags = cv2.CASCADE_SCALE_IMAGE)
    return rects
