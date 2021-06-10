#-*-coding: utf-8-*-  
import cv2
import numpy as np
import math
def stretch(img):
    max = float(img.max())
    min = float(img.min())
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = (255/(max-min))*img[i,j]-(255*min)/(max-min)         
    return img 
def dobinaryzation(img):
    max = float(img.max())
    min = float(img.min())
     
    x = max - ((max-min) / 2)
    ret, threshedimg = cv2.threshold(img, x, 255, cv2.THRESH_BINARY)
     
    return threshedimg

def find_retangle(contour):
	y, x = [], []
	
	for p in contour:
		y.append(p[0][0])
		x.append(p[0][1])
		
	return [min(y), min(x), max(y), max(x)]

def locate_license(img, orgimg):
	contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# 找出最大的三個區域
	blocks = []
	for c in contours:
		# 找出輪廓的左上點和右下點，由此計算它的面積和長寬比
		r = find_retangle(c)
		a = (r[2]-r[0]) * (r[3]-r[1])
		s = (r[2]-r[0]) / (r[3]-r[1])
		
		blocks.append([r, a, s])
		
	# 選出面積最大的3個區域
	blocks = sorted(blocks, key=lambda b: b[2])[-3:]
	
	# 使用顏色識別判斷找出最像車牌的區域
	maxweight, maxinedx = 0, -1
	for i in range(len(blocks)):
		b = orgimg[blocks[i][0][1]:blocks[i][0][3], blocks[i][0][0]:blocks[i][0][2]]
		# RGB轉HSV
		hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
		# 藍色車牌范圍
		lower = np.array([100,50,50])
		upper = np.array([140,255,255])
		# 根據閾值構建掩模
		mask = cv2.inRange(hsv, lower, upper)

		# 統計權值
		w1 = 0
		for m in mask:
			w1 += m / 255
		
		w2 = 0
		for w in w1:
			w2 += w
			
		# 選出最大權值的區域
		if w2 > maxweight:
			maxindex = i
			maxweight = w2
		
	return blocks[maxindex][0]
	
def find_license(img):
    img = cv2.resize(img,(1000,500))
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    '''
    拉伸灰
    '''
    stretchedimg = stretch(grayimg)
    '''
    開運算
    '''
    kernel = np.zeros((33, 43), dtype=np.uint8)
    cv2.circle(kernel, (16, 16),16, 1, -1)	 
    openingimg = cv2.morphologyEx(stretchedimg, cv2.MORPH_OPEN, kernel)
    strtimg = cv2.absdiff(stretchedimg,openingimg)
    '''
    將照片二值化
    '''
    binaryimg = dobinaryzation(strtimg)
    '''
    canny
    '''
    cannyimg = cv2.Canny(binaryimg, binaryimg.shape[0], binaryimg.shape[1])
	 
	#''' 消除小區域，保留大塊區域，從而定位車牌'''
	# 進行閉運算
    kernel = np.ones((5,19), np.uint8)
    closingimg = cv2.morphologyEx(cannyimg, cv2.MORPH_CLOSE, kernel)
	 
	# 進行開運算
    openingimg = cv2.morphologyEx(closingimg, cv2.MORPH_OPEN, kernel)
	 
	# 再次進行開運算
    kernel = np.ones((11,5), np.uint8)
    openingimg = cv2.morphologyEx(openingimg, cv2.MORPH_OPEN, kernel)

	# 消除小區域，定位車牌位置
    rect = locate_license(openingimg, img)
	
    return rect, img
	
if __name__ == '__main__':
    orgimg = cv2.imread('04.jpeg')
    orgimg = cv2.resize(orgimg,(1000,500))
    rect, img = find_license(orgimg)
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0,255,0),2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
