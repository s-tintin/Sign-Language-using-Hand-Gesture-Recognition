import numpy as np
import cv2
from matplotlib import pyplot as plt



indexes = ['A','B','C','D','E','F','I','K','L','N','R','S','T','U','V','W','X','Y']
for j in indexes:
	filename = "train/"+j+"/"
	
	i=1
	
	while(1):
		if i>3000:
			break
		imagename=filename+str(i)+".png"
		img1=cv2.imread(imagename,0)
		print(imagename)
		cv2.imshow('image',img1)

		img1 =cv2.GaussianBlur(img1,(5,5),0)


		r,t = cv2.threshold(img1,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


		kernel = np.ones((7,7),np.uint8)
		o = cv2.morphologyEx(t, cv2.MORPH_OPEN, kernel)
		
		savename=j+"/"+str(i)+".png"
		cv2.imwrite(savename,o)
		
		i=i+1
	print(filename)		
	

cv2.destroyAllWindows()
	



