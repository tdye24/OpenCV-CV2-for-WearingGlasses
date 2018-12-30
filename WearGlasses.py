import cv2 as cv
import numpy as np

pathj = 'D:\\MyProjects\\WearGlasses\\I.jpg'
pathg = 'D:\\MyProjects\\WearGlasses\\glasses.png'
pathf = 'D:\\MyProjects\\WearGlasses\\haarcascade_frontalface_default.xml'
pathe = 'D:\\MyProjects\\WearGlasses\\haarcascade_eye.xml'


def wear():
	glasses = cv.imread(pathg)
	face_cascade = cv.CascadeClassifier(pathf)
	eye_cascade = cv.CascadeClassifier(pathe)
	while True:
		centers = []
		cap = cv.VideoCapture(0)
		ret,img = cap.read()
		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.1, 3)
		for(x,y,w,h) in faces:
		    face_re = img[y:y+h, x:x+h]
		    face_re_g = gray[y:y+h, x:x+h]
		    eyes = eye_cascade.detectMultiScale(face_re_g)
		    for(ex,ey,ew,eh) in eyes:
		        cv.rectangle(face_re,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
		        centers.append((x+int(ex+0.5*ew),y+int(ey+0.5*eh),x + int(0.6*ex),y+ey))
		if len(centers) > 0:
		    eye_w = 2.0*abs(centers[1][0]-centers[0][0])
		    overlay_img = np.ones(img.shape,np.uint8)*0
		    gls_h,gls_w = glasses.shape[:2]
		    k = eye_w/gls_w
		    overlay_glasses = cv.resize(glasses,None,
		                                    fx = k,
		                                    fy = k,
		                                    interpolation = cv.INTER_AREA)
		    x = centers[0][0] if centers[0][0] < centers[1][0] else centers[1][0]
		    y = centers[0][1] if centers[0][1] < centers[1][1] else centers[1][1]
		    startx = centers[0][2] if centers[0][2] < centers[1][2] else centers[1][2]
		    starty = centers[0][3]
		    h,w = overlay_glasses.shape[:2]
		    overlay_img[starty:starty+h,startx:startx+w] = overlay_glasses

		    gray_glasses = cv.cvtColor(overlay_img,cv.COLOR_BGR2GRAY)
		    ret, mask = cv.threshold(gray_glasses,110,255,cv.THRESH_BINARY)
		    mask_inv = cv.bitwise_not(mask)
		    finalImg = cv.bitwise_and(img,img,mask=mask_inv)
		    cv.imshow("Wear =|=",finalImg)
		    if cv.waitKey(10) == 27:
		    	break
	cap.release()
	cv.destroyAllWindows()

if __name__ == '__main__':
	wear()
