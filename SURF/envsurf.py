

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras.layers import Input
from keras.preprocessing import image
import os
import numpy as np
import cv2
import pickle

cap = cv2.VideoCapture(0)
i=0
top, right, bottom, left = 10, 350, 225, 590


i=0
while(i<30):
	
	print(i)
	ret, img = cap.read()
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	
	
	cv2.rectangle(img,(left,top),(right,bottom),(0,0,0),1)

	img = cv2.GaussianBlur(img, (3, 3), 0)
	roi = img[top:bottom, right:left]
	cv2.imshow("image",img)
	cv2.imshow("roi",roi)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		imagename = "theimage"+str(i)+".png"
		cv2.imwrite(imagename,roi)
		i=i+1
		
















