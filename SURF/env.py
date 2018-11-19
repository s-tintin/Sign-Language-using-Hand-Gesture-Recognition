
import pickle
import os
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
i=0
top, right, bottom, left = 10, 350, 225, 590


clf = pickle.load(open("mlp_model.pkl", 'rb'))  
cluster_model = pickle.load(open("cluster_model.pkl",'rb'))

print("Loaded model from disk")


i=0
while(1):
	i=i+1
	ret, img = cap.read()
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	
	
	cv2.rectangle(img,(left,top),(right,bottom),(0,0,0),1)

	roi = img[top:bottom, right:left]

	s1 = cv2.resize(img,(700,700))
	s2 = cv2.resize(roi,(700,700))
	imstack = np.hstack((s1,s2))

	nimg = np.zeros((700,1400,3),np.uint8)
	print("black")
	font = cv2.FONT_HERSHEY_SIMPLEX
	
	frame = roi
	frame = cv2.resize(frame,(128,128))
	converted2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert from RGB to HSV
	#cv2.imshow("original",converted2)

	lowerBoundary = np.array([0,40,30],dtype="uint8")
	upperBoundary = np.array([43,255,254],dtype="uint8")
	skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)
	skinMask = cv2.addWeighted(skinMask,0.5,skinMask,0.5,0.0)
	#cv2.imshow("masked",skinMask)
    
    	skinMask = cv2.medianBlur(skinMask, 5)
    
    	skin = cv2.bitwise_and(converted2, converted2, mask = skinMask)
    	#frame = cv2.addWeighted(frame,1.5,skin,-0.5,0)
    	#skin = cv2.bitwise_and(frame, frame, mask = skinMask)

    	#skinGray=cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    
    	#cv2.imshow("masked2",skin)
    	img2 = cv2.Canny(skin,60,60)
    	#cv2.imshow("edge detection",img2)
    
    	''' 
    	hog = cv2.HOGDescriptor()
   	h = hog.compute(img2)
    	print(len(h))
    
    	'''
    	surf = cv2.xfeatures2d.SURF_create()
    	#surf.extended=True
    	img2 = cv2.resize(img2,(256,256))
    	kp, des = surf.detectAndCompute(img2,None)
    	#print(len(des))
    	#img2 = cv2.drawKeypoints(img2,kp,None,(0,0,255),4)
    	#plt.imshow(img2),plt.show()
    
    	#cv2.waitKey(0)
    	#cv2.destroyAllWindows()

	#print(len(des))
    	print(des)

	des = cluster_model.predict(des)

	X = np.array([np.bincount(des, minlength=150)])

	y_predict = clf.predict(X)
	indexes= ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']

	a=int(y_predict)
	print(a)
	letter=indexes[a]

	#cv2.putText(nimg,letter,(10,50), font, 1,(255,255,255),2,cv2.LINE_AA)
	cv2.putText(nimg,letter,(10,50), font, 1,(255,255,255),2,cv2.LINE_AA)
	
	print(nimg.shape,imstack.shape)
	imstack = np.vstack((imstack,nimg))

	cv2.imshow("stack",imstack)
	
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break

















