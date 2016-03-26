import cv2
import numpy as np
import argparse
from imutils import paths

arg_parse = argparse.ArgumentParser()
arg_parse.add_argument("-i" , "--images" , required = True , help = "path to images directory")
args = vars(arg_parse.parse_args())

for path in paths.list_images(args["images"]):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    features = cv2.cornerHarris(gray,2,3,0.04)
    
