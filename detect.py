from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i","--images",required = True, help = "path to image directory")
args = vars(ap.parse_args())

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for imagePath in paths.list_images(args["images"]):
    image = cv2.imread(imagePath)
    image = imutils.resize(image,width = min(400,image.shape[1]))
    orig = image.copy()

    #detect people
    (rects, weights) = hog.detectMultiScale(image,winStride=(4,4) ,
                                            padding = (8,8) , scale = 1.05)

    #original bounding boxes
    for(x,y,w,h) in rects:
        cv2.rectangle(orig,(x,y) , ((x+w) , (y+h)) , (0,0,255) , 2)

    #non maxima suppression for overlapping bounding boxes

    rects = np.array([ [x, y , x + w , y + h] for (x , y , w , h) in rects ])
    pick = non_max_suppression(rects, probs = None , overlapThresh= 0.65)

    #final bounding boxes
    for(xA , yA , xB , yB) in pick:
        cv2.rectangle(image , (xA , yA) , (xB , yB) , (0,255,0) , 2)

    cv2.imshow("image",image)
    cv2.waitKey(0)

