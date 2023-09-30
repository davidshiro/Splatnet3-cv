import numpy as np
import imutils
import glob
import cv2
import pytesseract 

from team_detect import TeamDetector
import sputil

#targeted charectars: 
#ĄąĆćĊċČčĎďĒēĖėĘęěĚĞğĠġĢģĦħĪīĮįİıĲĳĶķАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЫЪЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюя
#！＂＃＄％＆＇（）＊＋，－．／０１２３４５６７８９：；＜＝＞？＠ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ［＼］＾＿｀ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏpqrstuvwxyz｛｜｝～ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝﾞ¡¢£¤¥¦§¨©ª«¬­®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìîíïðñòóôõö÷øùúûüýāĂăÿĀ

class Player:
	#player class handles all ocr function
	def __init__(self, nameImg, paintImg, splatsImg, deathsImg, specialsImg, weaponImg, teamD) -> None:
		#use tesseract to ocr approximate name, then query against roster list
		nameImgRGB = cv2.cvtColor(nameImg, cv2.COLOR_BGR2RGB)
		self.td = teamD
		self.raw_name = pytesseract.image_to_string(self.name_prepro(nameImgRGB))
		self.name_data = self.td.name_search(self.raw_name)
		#run weapon ocr
		self.weapon = sputil.wep_detect(weaponImg)
		#run number ocr
		self.splats = self.ocr_num_splat(self.prepro(cv2.cvtColor(splatsImg, cv2.COLOR_BGR2GRAY)))
		self.assists = self.ocr_num_splat(cv2.cvtColor(self.assist_prepro(splatsImg), cv2.COLOR_BGR2GRAY))
		self.deaths = self.ocr_num_splat(self.prepro(cv2.cvtColor(deathsImg, cv2.COLOR_BGR2GRAY)))
		self.specials = self.ocr_num_splat(self.prepro(cv2.cvtColor(specialsImg, cv2.COLOR_BGR2GRAY)))
		self.paint = self.ocr_num_splat(self.prepro(cv2.cvtColor(paintImg, cv2.COLOR_BGR2GRAY)))

	#best guess as to what team player is on
	def cur_team(self):
		return self.name_data[0]
	
	#returns raw name data
	def name_d(self):
		return self.name_data
	
	#used when team name is confirmed
	def confirm_team(self, team_name):
		self.name_data = self.td.name_search_team(self.raw_name, team_name)

	def display_data(self):
		print(f"Name: {self.name_data[3]}")
		print(self.weapon)
		print(f"Splats: {self.splats}")
		print(f"Assists: {self.assists}")
		print(f"Deaths: {self.deaths}")
		print(f"Specials: {self.specials}")
		print(f"Paint: {self.paint}")

	#preprocessing functions to make images friendlier to CV; gives control over binarization and creates a border of white pixels
	def prepro(self, img):
		ret,out = cv2.threshold(img,120,255,cv2.THRESH_BINARY_INV)
		out = cv2.copyMakeBorder(out, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[255, 255, 255])
		return out
	
	def name_prepro(self, img):
		ret,out = cv2.threshold(img,150,255,cv2.THRESH_BINARY_INV)
		out = cv2.copyMakeBorder(out, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
		return out
	
	def assist_prepro(self, img):
		#first find only splats part of img
		ret,sp = cv2.threshold(img,170,255,cv2.THRESH_BINARY)
		#create "sloppy" mask to ensure area found will fully eliminate splats number
		sp = cv2.blur(sp, (4, 4))
		mask = cv2.inRange(sp, 0, 255)
		#find all numbers in image
		ret,assi = cv2.threshold(img,80,255,cv2.THRESH_BINARY)
		#mask out the splat number
		out = cv2.bitwise_and(assi, assi, mask = mask)
		#invert image (for consistancy)
		out = ~out
		out = cv2.copyMakeBorder(out, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[255, 255, 255])
		#resize assist text to match
		out = imutils.resize(out, width = int(out.shape[1] * 1.1875))
		#print(cv2.depth(out))
		#cv2.waitKey(0)
		return out
	
	def ocr_num_splat(self, img):
		digit_org = list()
		for templatePath in glob.glob("templates/numbers/*.png"):
			tem = cv2.imread(templatePath)
			tem = cv2.cvtColor(tem, cv2.COLOR_BGR2GRAY)
			(temH, temW) = tem.shape[:2]
			name = templatePath.replace("templates/numbers/",'').replace(".png",'')
			
			ret,pp_tem = cv2.threshold(tem,120,255,cv2.THRESH_BINARY)

			#cv2.imshow("number", pp_tem)
			#cv2.imshow("image", pp_img)
			#cv2.waitKey(0)

			res = cv2.matchTemplate(img, pp_tem, cv2.TM_CCOEFF)
			#(_, maxVal, _, _) = cv2.minMaxLoc(res)
			#print(f"max value of {name} is: {maxVal}")

			#find numbers with correlation threshold
			threshold = 2000000
			loc = np.where( res >= (threshold))
			org = np.array(list(zip(*loc[::-1])))
			org_full = sputil.non_max_suppression_fast_w_res(org, temW, temH, .3, res)
			if len(org) > 0:
				org = org_full[:,0:2]
			for orgY, orgX, val in org_full:
				newOrg = np.array([orgY,orgX])
				digit_org = digit_org + [[newOrg, val, name]]
			#print(f"digit_org as of {name}:{digit_org}")
		
		#considering all found numbers, remove overlapping numbers, only keeping ones with highest correlation
		pruned_org = sputil.non_max_suppression_fast_w_key(digit_org, temW, temH, .6)
		#print(f"pruned: {pruned_org}")
		drawn = img.copy()
		drawn = cv2.cvtColor(drawn, cv2.COLOR_GRAY2RGB)
		#cv2.imshow("ocrd", drawn)
		#cv2.waitKey(0)

		#parse result
		if len(pruned_org) == 0:
			return "0"
		output = ""
		for dig in pruned_org:
			if dig[2] != "pn" and dig[2] != "xn": output += dig[2].replace('n', '')

			#uncomment to display result on image
			#org = np.array([np.int64(dig[0][0]),np.int64(dig[0][1])])
			#print(dig[2])
			#print(org)
			#drawn = cv2.putText(drawn, str(dig[2].replace('n', '')), (org[0], org[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
			#cv2.rectangle(drawn, org, (org[0] + temW-2, org[1] + temH-2), (0,0,255), 1)
			#cv2.imshow("ocrd", drawn)
			#cv2.waitKey(0)
		#print(f"output: {output}")
		return output