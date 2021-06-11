#-*-coding: utf-8-*-  
import cv2
import numpy as np
import math
'''
拉伸灰
'''
def stretch(img):
    max = float(img.max())
    min = float(img.min())
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = (255/(max-min))*img[i,j]-(255*min)/(max-min)         
    return img
'''
將照片二值化
'''
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
    blocks = []
    for c in contours:
        r = find_retangle(c)
        a = (r[2]-r[0]) * (r[3]-r[1])
        s = (r[2]-r[0]) / (r[3]-r[1])		
        blocks.append([r, a, s])
    blocks = sorted(blocks, key=lambda b: b[2])[-3:]
    '''
    資料說：將 rgb 彩色的車牌照片轉換至 hsv 空間，在 hsv 空間裡，車牌是藍色的，所以用藍色識別出車牌區域
    '''
    maxweight, maxinedx = 0, -1
    for i in range(len(blocks)):
        b = orgimg[blocks[i][0][1]:blocks[i][0][3], blocks[i][0][0]:blocks[i][0][2]]
        hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
        lower = np.array([100,50,50])
        upper = np.array([140,255,255])
        '''
        創造 roi
        '''
        mask = cv2.inRange(hsv, lower, upper)
        w1 = 0
        for m in mask:
            w1 += m / 255
        w2 = 0
        for w in w1:
            w2 += w
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
    開閉開開閉
    創造先高在寬的全部都是零的 mask
    '''
    kernel = np.zeros((33, 34), dtype=np.uint8)
    cv2.circle(kernel, (16, 16),16, 1, -1)	 
    '''
    第一次開運算
    '''
    openingimg = cv2.morphologyEx(stretchedimg, cv2.MORPH_OPEN, kernel)
    '''
    獲取差分圖，這樣可以去除照片中的雜訊
    '''
    strtimg = cv2.absdiff(stretchedimg,openingimg)
    '''
    將照片二值化
    '''
    binaryimg = dobinaryzation(strtimg)
    '''
    canny
    '''
    cannyimg = cv2.Canny(binaryimg, binaryimg.shape[0], binaryimg.shape[1])
    '''
    消除小區域，保留大塊區域，從而定位車牌
    創造先高在寬的全部都是一的 mask
    '''
    kernel = np.ones((5,19), np.uint8)
    '''
    第一次閉運算
    '''
    closingimg = cv2.morphologyEx(cannyimg, cv2.MORPH_CLOSE, kernel)
    '''
    第二次開運算
    '''
    openingimg = cv2.morphologyEx(closingimg, cv2.MORPH_OPEN, kernel)
    '''
    第三次開運算
    '''
    kernel = np.ones((11,5), np.uint8)
    openingimg = cv2.morphologyEx(openingimg, cv2.MORPH_OPEN, kernel)
    '''
    第一次閉運算
    '''
    kernel = np.ones((5, 19), np.uint8)
    closingimg1 = cv2.morphologyEx(openingimg, cv2.MORPH_CLOSE, kernel)
    '''
    膨脹
    '''
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_dilated = cv2.dilate(closingimg1, kernel_2)
    rect = locate_license(kernel_dilated, img)
    return rect, img
if __name__ == '__main__':
    orgimg = cv2.imread('04.jpeg')
    orgimg = cv2.resize(orgimg,(1000,500))
    rect, img = find_license(orgimg)
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0,255,0),2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
