import numpy as np
import cv2

cap = cv2.VideoCapture('1.mp4')

# setting up a kernal for morphology
kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

# create a MoG background subtractor with 300 as a length of the history
fgbg = cv2.createBackgroundSubtractorMOG2(300)
# setting up the protion of the background model
fgbg.setBackgroundRatio(0.4)
# setting up the number of MoG
fgbg.setNMixtures(3)

while(1):
    # read the next frame
    ret, frame = cap.read()
    # get foreground mask
    fgmask = fgbg.apply(frame)
    # apply morpholoygy on the foreground mask to get a better result
    fgmask_ = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernal)
    # extract the background
    bg = fgbg.getBackgroundImage(frame)
    # show the current background
    cv2.imshow("background", bg)

    # check if Esc is presed exit the video
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destoryAllWindows()
