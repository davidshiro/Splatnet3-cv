import numpy as np
import imutils
import glob
import cv2
import json

def sort_orgFull_by_x(e):
	return e[0][0]

#removes overlapping boxes
def non_max_suppression_fast(org, width, height, overlapThresh):
	# if there are no boxes, return an empty list
	if org.size == 0:
		return []

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	# initialize the list of picked indexes
	pick = []	
	# grab the coordinates of the bounding boxes
	x1 = org[:,0]
	y1 = org[:,1]
	x2 = x1 + width
	y2 = y1 + height	
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)	
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)	
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])	
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)	
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]	
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))	

	# return only the bounding boxes that were picked using the
	# integer data type
	return org[pick]

#removes overlapping boxes (picks which to keep by)
def non_max_suppression_fast_w_res(org, width, height, overlapThresh, res):
	# if there are no boxes, return an empty list
	if org.size == 0:
		return []

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	# initialize the list of picked indexes
	pick = []	
	# grab the coordinates of the bounding boxes
	x1 = org[:,0]
	y1 = org[:,1]
	x2 = x1 + width
	y2 = y1 + height	
	# compute the area of the bounding boxes and sort the bounding
	# boxes by most likely to match
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	#print(f"res[y1,x1]: {res[y1,x1]}")
	idxs = np.argsort(res[y1,x1])	
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]	
		#print(f"Starting with: {idxs}")
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])	
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)	
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]	
		# delete all indexes from the index list that have

		
		workList = np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
		#print(f"Found boxes: {workList}")
		maxMatch = idxs[workList[0]]
		for idx in idxs[workList]:
			maxOrg = org[maxMatch]
			curOrg = org[idx]
			if res[maxOrg[1]][maxOrg[0]] < res[curOrg[1]][curOrg[0]]:
				maxMatch = idx
			#print(org[idx])
		#print(f"max index found: {maxMatch} with value {res[org[maxMatch][1]][org[maxMatch][0]]}")
		pick.append(maxMatch)
		#print(f"Found boxes: {idxs[workList]}")
		idxs = np.delete(idxs, workList)
		#print(f"Updated working list: {idxs}")	

	# return only the bounding boxes that were picked using the
	# integer data type
	ret = np.empty([len(pick),3])
	for i, p in enumerate(org[pick]):
		ret[i] = np.append(p, int(res[p[1]][p[0]]))
	return ret

def non_max_suppression_fast_w_key(org_full, width, height, overlapThresh):

	#extract org and val from org_full
	org = np.empty([len(org_full),2])
	val = np.empty(len(org_full))
	for i, orig in enumerate(org_full):
		org[i] = orig[0]
		val[i] = orig[1]
	# if there are no boxes, return an empty list
	if org.size == 0:
		return []

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	# initialize the list of picked indexes
	pick = []	
	# grab the coordinates of the bounding boxes
	x1 = org[:,0]
	y1 = org[:,1]
	x2 = x1 + width
	y2 = y1 + height	
	# compute the area of the bounding boxes and sort the bounding
	# boxes by most likely to match a nu
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(val)	
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]	
		#print(f"Starting with: {idxs}")
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])	
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		#h = np.maximum(0, yy2 - yy1 + 1)	
		# compute the ratio of horizontal overlap
		overlap = w / (x2[idxs[:last]]- x1[idxs[:last]])	
		# delete all indexes from the index list that have

		
		workList = np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
		maxMatch = idxs[workList[0]]
		for idx in idxs[workList]:
			maxOrg = org_full[maxMatch][1]
			curOrg = org_full[idx][1]
			if maxOrg < curOrg:
				maxMatch = idx
			#print(org_full[idx])
		#print(f"max index found: {org_full[maxMatch]} with value {maxOrg}")
		pick.append(maxMatch)
		#print(f"Found boxes: {idxs[workList]}")
		idxs = np.delete(idxs, workList)
		#print(f"Updated working list: {idxs}")	

	# return only the bounding boxes that were picked using the
	# integer data type
	ret = []
	for p in pick:
		ret.append(org_full[p])
	ret.sort(key=sort_orgFull_by_x)
	return ret


def wep_detect(crop):
	#conv to grayscale
	crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
	#cv2.imshow("Source", crop)
	max = None
	for imagePath in glob.glob("weapon_flat/*.png"):
		img = cv2.imread(imagePath)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = imutils.resize(img, width = 58)
		img = img[10:50,10:50]
		#img = cv2.Canny(img, 50, 200)
		maxVal = cv2.matchTemplate(img, crop, cv2.TM_CCOEFF_NORMED).max()
		if max is None or maxVal > max[0]:
			max = (maxVal, imagePath, img)
	wname = conv_wep_name(max[1])
	#print(f"Best guess is {wname}.\n")
	#cv2.imshow("Best guess", max[2])
	return wname

def conv_wep_name(path):
	wt = open("wep_name.json")
	wep_table = json.load(wt)
	#print(f"Converted {path} to")
	path = path.replace("weapon_flat\\Path_Wst_",'').replace(".png",'')
	result = wep_table["WeaponName_Table"][path]
	#print(result)
	wt.close()
	return result
