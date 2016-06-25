from aodHelper import *
from human_detection import is_human


class AbandonedObjectDetection:

    def __init__(self, cap, background, history=300, T=0.4, nMixtures=3,
        longBackgroundInterval=20, shortBackgroundINterval=1,
        k=7, maxe=2000, thh=80):

        self.cap = cap
        # background model
        self.BG = background
        self.BL = None
        self.BS = None

        # setting up a kernal for morphology
        self.kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

        # MoG for long background model
        self.fgbgl = MOG2init(history, T, nMixtures)
        # MoG for short background model
        self.fgbgs = MOG2init(history, T, nMixtures)

        self.longBackgroundInterval = longBackgroundInterval
        self.shortBackgroundINterval = shortBackgroundINterval

        self.clfg = longBackgroundInterval   # counter for longBackgroundInterval
        self.csfg = shortBackgroundINterval  # counter for shortBackgroundInteral

        # static obj likelihood
        self.L = np.zeros(np.shape(self.cap.read()[1])[0:2])

        self.static_obj_map = np.zeros(np.shape(self.cap.read()[1])[0:2])

        # static obj likelihood constants
        self.k, self.maxe, self.thh= k, maxe, thh

        # obj-extraction constants
        self.slidewindowtime = 0
        self.static_objs = []
        self.th_sp = 20**2 # a th for number of static pixels


    def get_abandoned_objs(self, frame):
        f2 = frame.copy()

        if self.clfg == self.longBackgroundInterval:
            frameL = np.copy(frame)
            self.fgbgl.apply(frameL)
            self.BL = self.fgbgl.getBackgroundImage(frameL)
            self.clfg = 0
        else:
            self.clfg += 1

        if self.csfg == self.shortBackgroundINterval:
            frameS = np.copy(frame)
            self.fgbgs.apply(frameS)
            self.BS = self.fgbgs.getBackgroundImage(frameS)
            self.csfg = 0
        else:
            self.csfg += 1

        # update short&long foregrounds
        FL = getForegroundMask(frame, self.BL, 70)
        FS = getForegroundMask(frame, self.BS, 70)
        FG = getForegroundMask(frame, self.BG, 70)

        # detec static pixels and apply morphology on it
        static = FL&cv2.bitwise_not(FS)&FG
        static = cv2.morphologyEx(static, cv2.MORPH_CLOSE, self.kernal)
        # dectec non static objectes and apply morphology on it
        not_static = FS|cv2.bitwise_not(FL)
        not_static = cv2.morphologyEx(not_static, cv2.MORPH_CLOSE, self.kernal)

        # update static obj likelihood
        self.L = (static == 255) * (self.L+1) + ((static == 255)^1) * self.L
        self.L = (not_static == 255) * (self.L-self.k) + ((not_static == 255)^1) * self.L
        self.L[ self.L>self.maxe ] = self.maxe
        self.L[ self.L<0 ] = 0

        # update static obj map
        self.static_obj_map[self.L >= self.thh ] = 254
        self.static_obj_map[self.L < self.thh ] = 0

        # if number of nonzero elements in static obj map greater than min window size squared there
        # could be a potential static obj, we will need to wait 200 frame to be pased if the condtion
        # still true we will call "extract_objs" function and try to find these objects.
        if(np.count_nonzero(clean_map(self.static_obj_map, self.static_objs)) > self.th_sp ):
            if(self.slidewindowtime > 200):
                #new_objs = extract_objs2(clean_map(static_obj_map, static_objs))
                new_objs = extract_objs2(self.static_obj_map)
                # if we get new object, first we make sure that they are not dublicated ones and then
                # put the unique static objects in "static_objs" variable
                if(new_objs):
                    for i in range(0, len(new_objs)):
                        if new_objs[i] not in self.static_objs:
                            self.static_objs.append(new_objs[i])
                self.slidewindowtime = 0
            else:
                self.slidewindowtime += 1
        else:
                self.slidewindowtime = 0 if self.slidewindowtime < 0 else self.slidewindowtime - 1
        # draw recatngle around static obj/s
        c=0
        for i in range (0, len(self.static_objs)):
            if(self.static_objs[i-c]):
                x, y = self.static_objs[i-c][0], self.static_objs[i-c][1]
                w, h = self.static_objs[i-c][2], self.static_objs[i-c][3]
                check_human_flag = self.static_objs[i-c][4]
                # check if the current static obj still in the scene 
                cv2.imshow("t", frame[y:y+h, x:x+w])
                cv2.imwrite("test_img.jpg", frame[y:y+h, x:x+w])
                if((np.count_nonzero(self.static_obj_map[y:y+h, x:x+w]) < w * h * .1)):
                    self.static_objs.remove(self.static_objs[i-c])
                    c += 1
                    continue
                if(check_human_flag):
                    if(check_human_flag > 25): # check if the founded obj is a human ever 1 sec
                        self.static_objs[i-c][4] = 0
                        if(is_human(frame[y:y+h, x:x+w])):
                            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                            continue
                    else:
                        self.static_objs[i-c][4] += 1
                # cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
        return self.static_objs

if __name__ == '__main__':
    """
    Run a demo.
    """
    import cv2

    from AbandonedObjectDetection import *

    cap = cv2.VideoCapture('2.mp4')
    BG = cv2.imread('bg.jpg')

    aod = AbandonedObjectDetection(cap, BG)

    while (1):
        _,frame = cap.read()
        objs = aod.get_abandoned_objs(frame)

        for obj in objs:
            x,y,w,h,_=obj
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)

        cv2.imshow("1",frame)
        
        key = cv2.waitKey(25) & 0xff
        if key == 27:
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()
