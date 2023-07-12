import numpy as np
import imutils
import glob
import cv2

import sputil
from player import Player

def sortByY(e):
	return e[1]

def vconcat_resize(img_list, interpolation 
                   = cv2.INTER_CUBIC):
      # take minimum width
    w_min = max(img.shape[1] 
                for img in img_list)
      
    # resizing images
    im_list_resize = [cv2.resize(img,
                      (w_min, int(img.shape[0] * w_min / img.shape[1])),
                                 interpolation = interpolation)
                      for img in img_list]
    # return final image
    return cv2.vconcat(im_list_resize)

class ImgToStat:
	def __init__(self, imgsrc, pTem) -> None:
		self.playerList = list()

		self.primaryTem = cv2.cvtColor(pTem, cv2.COLOR_BGR2GRAY)
		self.primaryTem = cv2.Canny(self.primaryTem, 50, 200)

		self.image = imgsrc
		(self.r, self.vicXAnchor, self.resized) = self.calibrate_scale()
		self.create_players()
		self.disp_debug()

	def disp_debug(self):
		cv2.imshow("debug", imutils.resize(self.image, height=800))
		cv2.waitKey(0)

	def disp_resized(self):
		cv2.imshow("base resized", self.resized)

	def calibrate_scale(self):
		(tH, tW) = self.primaryTem.shape[:2]
		gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		found = None
		# loop over the scales of the image
		for scale in np.linspace(0.2, 1.0, 50)[::-1]:
			# resize the image according to the scale, and keep track
			# of the ratio of the resizing
			resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
			r = gray.shape[1] / float(resized.shape[1])
			# if the resized image is smaller than the template, then break
			# from the loop
			if resized.shape[0] < tH or resized.shape[1] < tW:
				break
			# detect edges in the resized, grayscale image and apply template
			# matching to find the template in the image
			edged = cv2.Canny(resized, 50, 200)
			result = cv2.matchTemplate(edged, self.primaryTem, cv2.TM_CCOEFF)
			(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
			# if we have found a new maximum correlation value, then update
			# the bookkeeping variable
			if found is None or maxVal > found[0]:
				found = (maxVal, maxLoc, r, result)
		# unpack the bookkeeping variable and compute the (x, y) coordinates
		# of the bounding box based on the resized ratio
		(maxVal, _, scale, res) = found
		rescaled = imutils.resize(self.image, width = int(self.image.shape[1]/scale))
		resize = rescaled #copy of rescale image for internal use
		threshold = .6
		loc = np.where( res >= (threshold*maxVal))
		count = 0
		org = np.array(list(zip(*loc[::-1])))
		org = sputil.non_max_suppression_fast(org, tW, tH, .5)
		for pt in list(org)[::-1]:
			count += 1
			print(f"Primary object located. {count}")
			cv2.rectangle(rescaled, pt, (pt[0] + tW, pt[1] + tH), (0,0,255), 1)
			rescaled = cv2.putText(rescaled, str(count), (pt[0], pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
			victoryAnc = pt[0]
		return (scale, victoryAnc, resize)

	def create_players(self):
		#locates splat icons
		splat_org = list()
		for templatePath in glob.glob("templates/splats/*.PNG"):
			tem = cv2.imread(templatePath)
			tem = cv2.cvtColor(tem, cv2.COLOR_BGR2GRAY)
			tem = cv2.Canny(tem, 50, 200)
			(temH, temW) = tem.shape[:2]

			edged = cv2.Canny(self.resized, 50, 200)
			res = cv2.matchTemplate(edged, tem, cv2.TM_CCOEFF)
			(_, maxVal, _, _) = cv2.minMaxLoc(res)

			threshold = .6
			loc = np.where( res >= (threshold*maxVal))
			count = 0
			org = np.array(list(zip(*loc[::-1])))
			org = sputil.non_max_suppression_fast(org, temW, temH, .5)
			splat_org = splat_org + list(org)

		#locates death icons
		death_org = list()
		for templatePath in glob.glob("templates/deaths/*.PNG"):
			tem = cv2.imread(templatePath)
			tem = cv2.cvtColor(tem, cv2.COLOR_BGR2GRAY)
			tem = cv2.Canny(tem, 50, 200)
			(temH, temW) = tem.shape[:2]

			edged = cv2.Canny(self.resized, 50, 200)
			res = cv2.matchTemplate(edged, tem, cv2.TM_CCOEFF)
			(_, maxVal, _, _) = cv2.minMaxLoc(res)

			threshold = .6
			loc = np.where( res >= (threshold*maxVal))
			count = 0
			org = np.array(list(zip(*loc[::-1])))
			org = sputil.non_max_suppression_fast(org, temW, temH, .5)
			death_org = death_org + list(org)

		#locate data for each player
		splat_org.sort(key=sortByY)
		death_org.sort(key=sortByY)
		pl = []
		for i in range(8):
			pt = list(splat_org)[i]
			paint = self.resized[(pt[1]+28):(pt[1]+56), (pt[0]-95):(pt[0]-22)]
			splats = self.resized[(pt[1]+28):(pt[1]+54),(pt[0]-9):(pt[0]+61)]
			#paint = self.image[int(self.r*(pt[1]+28)):int(self.r*(pt[1]+56)), int(self.r*(pt[0]-95)):int(self.r*(pt[0]-22))]
			#splats = self.image[int(self.r*(pt[1]+28)):int(self.r*(pt[1]+54)),int(self.r*(pt[0]-9)):int(self.r*(pt[0]+61))]
			wep = self.resized[(pt[1]-4):(pt[1]+61),(self.vicXAnchor+2):(self.vicXAnchor+67)]
			name = self.resized[(pt[1]+28):(pt[1]+54), (self.vicXAnchor+75):(self.vicXAnchor+275)]
			pt = list(death_org)[i]
			deaths = self.resized[(pt[1]+27):(pt[1]+53),(pt[0]):(pt[0]+50)]
			specials = self.resized[(pt[1]+27):(pt[1]+53),(pt[0]+45):(pt[0]+86)]
			#cv2.imshow("playerinfo", vconcat_resize([name,wep,splats,deaths,specials,paint]))
			pl.append(Player(name, paint, splats, deaths, specials, wep))
			#cv2.waitKey(0)

	