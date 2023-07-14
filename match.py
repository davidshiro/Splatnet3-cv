import numpy as np
import argparse
import glob
import cv2

from img_to_stat import ImgToStat

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="Path to images where template will be matched")
args = vars(ap.parse_args())

def sortByY(e):
	return e[1]

# load the image image, convert it to grayscale, and detect edges
template = cv2.imread("templates/victory.PNG")
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]
#cv2.imshow("Template", template)

# loop over the images to find the template in
for imagePath in glob.glob(args["images"] + "/*.jpg"):
	image = cv2.imread(imagePath)
	stats = ImgToStat(image, cv2.imread("templates/victory.PNG"))

for imagePath in glob.glob(args["images"] + "/*.png"):
	image = cv2.imread(imagePath)
	print(imagePath)
	stats = ImgToStat(image, cv2.imread("templates/victory.PNG"))

