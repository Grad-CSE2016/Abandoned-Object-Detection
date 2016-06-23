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
    #cv2.imshow('arr', arr)
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
