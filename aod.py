import numpy as np
import cv2

def getForegroundMask(frame, background, th):
    # reduce the nois in the farme
    frame =  cv2.blur(frame, (5,5))
    # get the absolute difference between the foreground and the background
    fgmask= cv2.absdiff(frame, background)
    # convert foreground mask to gray
    fgmask = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)
    # apply threshold (th) on the foreground mask
    _, fgmask = cv2.threshold(fgmask, th, 255, cv2.THRESH_BINARY)
    # setting up a kernal for morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # apply morpholoygy on the foreground mask to get a better result
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    return fgmask

def MOG2init(history, T, nMixtures):
    # create an instance of MoG and setting up its history length
    fgbg = cv2.createBackgroundSubtractorMOG2(history)
    # setting up the protion of the background model
    fgbg.setBackgroundRatio(T)
    # setting up the number of MoG
    fgbg.setNMixtures(nMixtures)
    return fgbg

cap = cv2.VideoCapture('1.mp4')

# setting up a kernal for morphology
kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

# MoG for long background model
fgbgl = MOG2init(300, 0.4, 3)
# MoG for short background model
fgbgs = MOG2init(300, 0.4, 3)

longBackgroundInterval = 20
shortBackgroundINterval = 1

clfg = longBackgroundInterval   # counter for longbackgroundInterval
csfg = shortBackgroundINterval  # counter for shortBackgroundInteral

# static obj likelihood
L = np.zeros(np.shape(cap.read()[1])[0:2])
static_obj = np.zeros(np.shape(cap.read()[1])[0:2])
k, maxe, thh= 100, 1000, 700

while(1):
    ret, frame = cap.read()

    if clfg == longBackgroundInterval:
        frameL = np.copy(frame)
        fgbgl.apply(frameL)
        BL = fgbgl.getBackgroundImage(frameL)
        clfg = 0
    else:
        clfg += 1

    if csfg == shortBackgroundINterval:
        frameS = np.copy(frame)
        fgbgs.apply(frameS)
        BS = fgbgs.getBackgroundImage(frameS)
        csfg = 0
    else:
        csfg += 1

    # update short&long foregrounds
    FL = getForegroundMask(frame, BL, 70)
    FS = getForegroundMask(frame, BS, 70)

    # detec static pixels and apply morphology on it
    static = FL&cv2.bitwise_not(FS)
    static = cv2.morphologyEx(static, cv2.MORPH_CLOSE, kernal)
    # dectec non static objectes and apply morphology on it
    not_static = FS|cv2.bitwise_not(FL)
    not_static = cv2.morphologyEx(not_static, cv2.MORPH_CLOSE, kernal)

    # update static obj likelihood
    L = (static == 255) * (L+1) + ((static == 255)^1) * L
    L = (not_static == 255) * (L-k) + ((not_static == 255)^1) * L
    L[ L>maxe ] = maxe
    L[ L<0 ] = 0

    static_obj[L >= thh ] = 255
    static_obj[L < thh ] = 0

    frame[L>=thh] = 0,0,255
    #cv2.imshow("static_obj", static_obj)
    cv2.imshow("frame", frame)

    # check if Esc is presed exit the video
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
cv2.destoryAllWindows()
