import cv2 as cv
import numpy as np
def load_image(path):
    src=cv.imread(path)
    '''
    1000*500 可以找到第十張
    '''
    src = cv.resize(src,(1000,500))
    '''
    可以找到第一張
    '''
    #src=cv.resize(src,(400,int(400 * src.shape[0] / src.shape[1])))
    return src
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
def gray_stretch(image):
    max_value=float(image.max())
    min_value=float(image.min())
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i,j]=(255/(max_value-min_value)*image[i,j]-(255*min_value)/(max_value-min_value))
    return image
def image_binary(image):
    max_value=float(image.max())
    min_value=float(image.min()) 
    ret=max_value-(max_value-min_value)/2
    ret,thresh=cv.threshold(image,ret,255,cv.THRESH_BINARY)
    return thresh
'''
矩形輪廓角點，尋找到矩形之後記錄角點，用來參考以及畫圖
'''
def find_rectangle(contour):
    y,x=[],[]
    for value in contour:
        y.append(value[0][0])
        x.append(value[0][1])
    return [min(y),min(x),max(y),max(x)]
'''
定位車牌函式，需要二照片當做函式引數，
一個用來找位置，找位置的照片為經過多次型態學操做的照片，
另一個為原圖用來繪製矩形，
此函式運用權值，實現了定位的最高概率
'''
def loacte_plate(image,after):
    contours,hierarchy=cv.findContours(image,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    img_copy = after.copy()
    solving=[]
    for c in contours:
        r=find_rectangle(c)
        '''
        算出面積、長寬比
        '''
        a=(r[2]-r[0])*(r[3]-r[1]) 
        s=(r[2]-r[0])/(r[3]-r[1])
        solving.append([r,a,s])
    '''
    調一：
    調二：
    '''
    solving=sorted(solving,key=lambda b: b[2])[-3:]
    '''
    資料說：將 rgb 彩色的車牌照片轉換至 hsv 空間，在 hsv 空間裡，車牌是藍色的，所以用藍色識別出車牌區域
    '''
    maxweight,maxindex=0,-1
    for i in range(len(solving)):
        wait_solve=after[solving[i][0][1]:solving[i][0][3],solving[i][0][0]:solving[i][0][2]]
        hsv=cv.cvtColor(wait_solve,cv.COLOR_BGR2HSV)
        lower=np.array([100,50,50])
        upper=np.array([140,255,255])
        '''
        運用 inrange() 找出 roi
        '''
        mask=cv.inRange(hsv,lower,upper)
        w1=0
        for m in mask:
            w1+=m/255
        w2=0
        for n in w1:
            w2+=n
        if w2>maxweight:
            maxindex=i
            maxweight=w2
    return solving[maxindex][0]  
def find_plates(image):
    gray_image=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    stretchedimage=gray_stretch(gray_image)
    kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE,(30,30))
    '''
    第一次開運算並獲取差分圖，這樣可以去除照片中的雜訊
    '''
    openingimage=cv.morphologyEx(stretchedimage,cv.MORPH_OPEN,kernel)
    strtimage=cv.absdiff(stretchedimage,openingimage)
    binaryimage=image_binary(strtimage)
    canny=cv.Canny(binaryimage,binaryimage.shape[0],binaryimage.shape[1])
    '''
    消除小區域，保留大塊區域，從而定位車牌
    創造先高在寬的全部都是一的 mask
    '''
    kernel=np.ones((5,24),np.uint8) # 5 24
    '''
    第一次閉運算
    '''
    closingimage=cv.morphologyEx(canny,cv.MORPH_CLOSE,kernel)
    '''
    第二次開運算
    '''
    openingimage=cv.morphologyEx(closingimage,cv.MORPH_OPEN,kernel)
    kernel=np.ones((11,6),np.uint8) # 11 6
    openingimage=cv.morphologyEx(openingimage,cv.MORPH_OPEN,kernel)
    '''
    第三次開運算
    '''
    rect=loacte_plate(openingimage,image)
    cv.imshow('image',image)
    cv.rectangle(image, (rect[0],rect[1]),(rect[2],rect[3]),(0, 255, 0), 2)
    cv.imshow('after', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
def runing(): 
    find_plates(load_image('10.jpeg'))
runing()
