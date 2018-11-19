
import pickle
import os
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
i=0
top, right, bottom, left = 10, 350, 225, 590



scalar,clf = pickle.load(open("tuple.pkl", 'rb'))  

print("Loaded model from disk")


i=0
while(1):
	i=i+1
	ret, img = cap.read()
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	
	
	cv2.rectangle(img,(left,top),(right,bottom),(0,0,0),1)

	roi = img[top:bottom, right:left]

	s1 = cv2.resize(img,(700,700))
	s2 = cv2.resize(roi,(700,700))
	imstack = np.hstack((s1,s2))

	nimg = np.zeros((700,1400),np.uint8)
	print("black")
	font = cv2.FONT_HERSHEY_SIMPLEX
	hm = cv2.HuMoments(cv2.moments(roi)).flatten()
	x=scalar.transform([hm])
	print(x)
	y_predict = clf.predict(x)
	indexes= ['A','B','C','D','E','F','I','K','L','N','R','S','T','U','V','W','X','Y','Z']

	a=int(y_predict)
	letter=indexes[a]

	cv2.putText(nimg,letter,(10,50), font, 1,(255,255,255),2,cv2.LINE_AA)

	
	imstack = np.vstack((imstack,nimg))

	cv2.imshow("stack",imstack)
	
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break

















