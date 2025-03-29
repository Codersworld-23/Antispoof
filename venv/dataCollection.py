from cvzone.FaceDetectionModule import FaceDetector
import cv2
import cvzone
from time import time
import os

TF_ENABLE_ONEDNN_OPTS=0
outputFolderPath = "Dataset/DataCollect"
offsetpercentageW = 10
offsetpercentageH = 20
confidence = .8
classID = 0  # 0 = fake : 1 = Real

save = True
debug = False
camwidth, camheight = 640, 480
blurrThreshold = 35  # larger is more focussed
# vediopath = "C:/Users/hp/Pictures/Camera Roll/me.mp4"
cap = cv2.VideoCapture(1)
cap.set(3, camwidth)
cap.set(4, camheight)
detector = FaceDetector()  # minDetectionCon=0.5, modelSelection=0

while True:       
    
    success, img = cap.read()       
    imgout = img.copy()
    img, bboxs = detector.findFaces(img,draw=False ) # 
    
    listBlur = [] # stores true and false value if the image is blur or not
    listInfo = [] # normalised value and class name for label text file

    if bboxs:            
        for bbox in bboxs:
            # bbox contains 'id', 'bbox', 'score', 'center'
            # ---- Get Data  ---- #            
            # center = bbox["center"]
            x, y, w, h = bbox['bbox']
            score = bbox["score"][0]
            
            if score > confidence : 
                
                # offset to face detected
                offsetW = (offsetpercentageW / 100) *w
                x = (int)( x - offsetW )
                w = (int)(w + offsetW*2 )            
                offsetH = (offsetpercentageH / 100) *h
                y = (int)( y - offsetH*3 )
                h = (int)(h + offsetH*3 )
                
                if x<0 :
                    x = 0
                    
                if y<0 :
                    y = 0
                    
                if w < 0:
                    w =0
                    
                if h < 0:
                    h = 0
                
                # find blurriness [more we stay stable more is the value]
                imgFace = img[y:y+h, x:x+w]
                cv2.imshow("Face", imgFace)            
                blurrValue =(int)(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                if (blurrValue > blurrThreshold):
                    listBlur.append(True)
                else:
                    listBlur.append(False)
                    
                #  Normalization
                ih, iw, _ = img.shape
                xc, yc = x+w/2, y+h/2
                xcn = round(xc/iw, 6)
                ycn = round(yc/ih, 6)
                wn = round(w/iw, 6)
                hn = round(h/ih, 6)
                
                if xcn > 1 :
                    xcn =  1
                    
                if ycn > 1 :
                    ycn =  1
                    
                if wn > 1:
                    wn = 1
                    
                if hn > 1:
                    hn =  1
                
                listInfo.append( f"{classID} {xcn} {ycn} {wn} {hn} \n")
                
                # Custom Geometry and text
                cv2.rectangle(imgout,(x,y,w,h),(255,0,0),3)
                cv2.putText(imgout,f'Score : {(int)(score*100)}% - Blurr : {blurrValue}', (x, y-20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0),  1)

                if debug:
                    cv2.rectangle(img,(x,y,w,h),(255,0,0),3)
                    cv2.putText(img,f'Score : {(int)(score*100)}% - Blurr : {blurrValue}', (x, y-20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0),  1)

        # SAving our image from the camera
        if(save):
            if all(listBlur) and listBlur != []:
                timenow = time()
                timenow = str(timenow).split(".")
                timenow = timenow[0]+timenow[1]
                cv2.imwrite(f"{outputFolderPath}/{timenow}.jpg",img)
                for info in listInfo:
                    f = open(f"{outputFolderPath}/{timenow}.txt",'a')
                    f.write(info)
                    f.close()
    cv2.imshow("Image", imgout)
    cv2.waitKey(1)

