import cv2
import numpy as np

while(1):
	text = raw_input("Enter text:")
	text=text.lower()
	print(str(text))
	text = list(text)
	print(text)
	f=1
	vis=None
	for i in text:
		print(i)
		if i.isalpha():
			imagename =i+".jpg"
			img = cv2.imread(imagename)
			img = cv2.resize(img, (150, 100)) 
			if f==1:
				vis=img
				f=0
			else:
				vis =np.concatenate((vis,img),axis=1)
		if i==' ':
			print("seen"+i+"space")
			img = np.zeros((100,20,3), np.uint8)
			vis =np.concatenate((vis,img),axis=1)
		
	cv2.imshow("image",vis)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
