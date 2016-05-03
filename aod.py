import numpy as np
import cv2
from human_detection import is_human

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

def extract_objs(image, step_size, window_size):
    # a threshold for min static pixels needed to be found in the sliding window
    th = (window_size**2) * 0.1
    current_nonzero_elements = 0
    # penalty is how meny times the expanding process didn't manage to find new
    # static pixels, step is how much the expanding of the sliding will be and objs is a returned
    # value containing the objects in the image
    penalty, step, objs = 0, 5, []
    # a while loop for sliding window in x&y
    y = 0
    while(y < image.shape[0]):
        x = 0
        while(x < image.shape[1]):
            # counting the nonzero elements in the current window
            current_nonzero_elements = np.count_nonzero(image[y:y+window_size, x:x+window_size])
            print(current_nonzero_elements, th)
            if(current_nonzero_elements > th):
                width =  window_size
                height = window_size
                # expand in x & y
                penalty = 0
                while(penalty < 1):
                    dx = np.count_nonzero(image[y:y+height, x+width:x+width+step])
                    dy = np.count_nonzero(image[y+height: y+height+step, x:x+width])
                    if(dx == 0 and dy == 0):
                        penalty += 1
                        width += step
                        height += step
                    elif(dx >= dy):
                        width += step
                    else:
                        height += step

                objs.append([x, y, width, height])
                y += height
                break
            x += step_size
        y += step_size
    if(len(objs)):
        return objs
    return

def extract_objs2(im, min_w=15, min_h=15, max_w=500, max_h=500):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    arr = cv2.dilate(im, kernel, iterations=2)
    arr = np.array(arr, dtype=np.uint8) 
    _, th = cv2.threshold(arr,127,255,0)
    im2, contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    objs = []
    cv2.imwrite("tmp2.jpg", arr)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if (w >= min_w) & (w < max_w) & (h >= min_h ) & (h < max_h):
            objs.append([x,y,w,h, 1]) # The last one means that it is still needed to check
                                      # if it is a human or an obj
        else:
            print(w,h)
    return objs

# this function returns static object map without pre-founded objects
def clean_map(m, o):
    rslt = np.copy(m)
    for i in range (0, len(o)):
        x, y= o[i][0], o[i][1]
        w, h= o[i][2], o[i][3]
        rslt[y:y+h, x:x+w] = 0
    return rslt

print(is_human(cv2.imread("test_img.jpg")))

#cap = cv2.VideoCapture('AVSS_AB_EVAL_divx.avi')
cap = cv2.VideoCapture('3.mp4')

# background model
BG = cv2.imread('bg.jpg')

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

static_obj_map = np.zeros(np.shape(cap.read()[1])[0:2])

# static obj likelihood constants
k, maxe, thh= 7, 2000, 800

# obj-extraction constants
slidewindowtime = 0
minwindowsize = 70
stepsize = 15
static_objs = []
th_sp = 20**2 # a th for number of static pixels

while(1):
    ret, frame = cap.read()
    f2 = frame.copy()
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
    FG = getForegroundMask(frame, BG, 70)

    # detec static pixels and apply morphology on it
    static = FL&cv2.bitwise_not(FS)&FG
    static = cv2.morphologyEx(static, cv2.MORPH_CLOSE, kernal)
    # dectec non static objectes and apply morphology on it
    not_static = FS|cv2.bitwise_not(FL)
    not_static = cv2.morphologyEx(not_static, cv2.MORPH_CLOSE, kernal)

    # update static obj likelihood
    L = (static == 255) * (L+1) + ((static == 255)^1) * L
    L = (not_static == 255) * (L-k) + ((not_static == 255)^1) * L
    L[ L>maxe ] = maxe
    L[ L<0 ] = 0

    # update static obj map
    static_obj_map[L >= thh ] = 254
    static_obj_map[L < thh ] = 0

    # if number of nonzero elements in static obj map greater than min window size squared there
    # could be a potential static obj, we will need to wait 200 frame to be pased if the condtion
    # still true we will call "extract_objs" function and try to find these objects.
    if(np.count_nonzero(clean_map(static_obj_map, static_objs)) > th_sp ):
        if(slidewindowtime > 200):
            new_objs = extract_objs2(clean_map(static_obj_map, static_objs))
            # if we get new object, first we make sure that they are not dublicated ones and then
            # put the unique static objects in "static_objs" variable
            if(new_objs):
                for i in range(0, len(new_objs)):
                    if new_objs[i] not in static_objs:
                        static_objs.append(new_objs[i])
            slidewindowtime = 0
        else:
            slidewindowtime += 1
    else:
            slidewindowtime = 0 if slidewindowtime < 0 else slidewindowtime - 1
    # draw recatngle around static obj/s
    c=0
    for i in range (0, len(static_objs)):
        if(static_objs[i-c]):
            x, y = static_objs[i-c][0], static_objs[i-c][1]
            w, h = static_objs[i-c][2], static_objs[i-c][3]
            check_human_flag = static_objs[i-c][4]
            # check if the current static obj still in the scene 
            cv2.imshow("t", frame[y:y+h, x:x+w])
            cv2.imwrite("test_img.jpg", frame[y:y+h, x:x+w])
            if((np.count_nonzero(static_obj_map[y:y+h, x:x+w]) < w * h * .1)):
                static_objs.remove(static_objs[i-c])
                c += 1
                continue
            if(check_human_flag):
                if(check_human_flag > 25): # check if the founded obj is a human ever 1 sec
                    static_objs[i-c][4] = 0
                    if(is_human(frame[y:y+h, x:x+w])):
                        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                        continue
                else:
                    static_objs[i-c][4] += 1
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)

    print(static_objs)
    cv2.imshow("frame", frame)
    cv2.imshow("BG", BS)
    #cv2.imshow("bgs", BS)
    #cv2.imshow("bgl", BL)
    #cv2.imshow("L", clean_map(static_obj_map, static_objs))
    # check if Esc is presed exit the video
    key = cv2.waitKey(1) & 0xff
    if key == 27:
        cv2.imwrite("f1.jpg", frame)
        cv2.imwrite("BG.jpg", BS)
        break


cap.release()
cv2.destoryAllWindows()
