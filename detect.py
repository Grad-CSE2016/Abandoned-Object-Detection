from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import datetime

def detect_face(faceCascadePath,image,scaleFactor = 1.1 , minNeighbors = 5 , minSize = (30,30)):
    faceCascade = cv2.CascadeClassifier(faceCascadePath)
    rects = faceCascade.detectMultiScale(image,scaleFactor = scaleFactor ,
    minNeighbors = minNeighbors , minSize = minSize , flags = cv2.CASCADE_SCALE_IMAGE)
    return rects

ap = argparse.ArgumentParser()
ap.add_argument("-f" , "--face" , required = True , help = "path to the face cascade")
ap.add_argument("-i","--image",required = True, help = "path to the input image")
args = vars(ap.parse_args())

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

image = cv2.imread(args["image"])
image = imutils.resize(image,width = min(400,image.shape[1]))
orig = image.copy()
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
isPerson = False

#start = datetime.datetime.now()
#detect person
(rects, weights) = hog.detectMultiScale(image,winStride=(8,8), padding = (8,8) , scale = 1.03)
#Check if a Person is detected if not try face detection
if type(rects) is not tuple and rects.size > 0:
    isPerson = True
else:
    rects = detect_face(args["face"],gray , scaleFactor = 1.1 , minNeighbors = 5 , minSize = (30,30))
    # return true if len(rects) bigger than 0
    if len(rects) > 0:
        isPerson = True
    else:
        isPerson = False
#print("[INFO] detection took: {}s".format((datetime.datetime.now() - start).total_seconds()))
print (isPerson)
