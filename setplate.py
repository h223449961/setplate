import cv2
import numpy as np
import math
'''
拉伸灰
以下是拉伸灰公式：
g(x,y) = 255 / (b - a) * [f(x,y) - a]
a = min[f(x,y)] 為最小灰
b = max[f(x,y)] 為最大灰
f(x,y) 為輸入圖象
g(x,y) 為輸出圖象
他可以有選擇地拉伸某段灰度區間，以改善輸出圖像
如果圖像的灰度集中在較暗的區域而導致圖像偏暗，可以用灰度拉伸功能來拉伸物體灰度區間以改善圖像
如果圖像的灰度集中在較亮的區域而導致圖像偏亮，可以用灰度拉伸功能來壓縮物體灰度區間以改善圖像
如果 a = 0 b = 255 則照片沒有什麼改變
'''
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
'''
矩形輪廓角點，尋找到矩形之後記錄角點，用來參考以及畫圖
'''
def find_retangle(contour):
    y, x = [], []
    for p in contour:
        y.append(p[0][0])
        x.append(p[0][1])
    return [min(y), min(x), max(y), max(x)]
'''
定位車牌函式，需要二照片當做函式引數，
一個用來找位置，找位置的照片為經過多次型態學操做的照片，
另一個為原圖用來繪製矩形，
此函式運用權值，實現了定位的最高概率
'''
def locate_license(img, orgimg):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blocks = []
    for c in contours:
        r = find_retangle(c)
        '''
        算出面積、長寬比
        '''
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
        運用 inrange() 找出 roi
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
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    stretchedimg = stretch(grayimg)
    '''
    開閉開開閉
    創造先高在寬的全部都是零的 mask
    '''
    kernel = np.zeros((33, 34), dtype=np.uint8)
    cv2.circle(kernel, (16, 16),16, 1, -1)	 
    '''
    第一次開運算並獲取差分圖，這樣可以去除照片中的雜訊
    '''
    openingimg = cv2.morphologyEx(stretchedimg, cv2.MORPH_OPEN, kernel)
    strtimg = cv2.absdiff(stretchedimg,openingimg)
    binaryimg = dobinaryzation(strtimg)
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
    orgimg = cv2.imread('08.jpeg')
    orgimg = cv2.resize(orgimg,(1000,500))
    rect, img = find_license(orgimg)
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0,255,0),2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
