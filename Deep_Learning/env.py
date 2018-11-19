

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

cap = cv2.VideoCapture(0)
i=0
top, right, bottom, left = 10, 350, 225, 590

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


i=0
while(1):
	i=i+1
	ret, img = cap.read()
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	
	
	cv2.rectangle(img,(left,top),(right,bottom),(0,0,0),1)

	img = cv2.GaussianBlur(img, (3, 3), 0)
	roi = img[top:bottom, right:left]

	cv2.imwrite("i.png",roi)

	s1 = cv2.resize(img,(300,300))
	s2 = cv2.resize(roi,(300,300))
	imstack = np.hstack((s1,s2))
	
	print(imstack.shape)

	nimg = np.zeros((300,600),np.uint8)
	print("black")
	print(nimg.shape)
	font = cv2.FONT_HERSHEY_SIMPLEX
	
	test_image = image.load_img('i.png', target_size = (64, 64))
	print ("Done")
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
	result = loaded_model.predict(test_image)
	#print(indices)
	print(result)
	print(result*100)
	#print('Predicted:', decode_predictions(result))
	k = np.argmax(result)
	print("max is\n"+str(k))

	show=str(result)+"\n"+str(k)
	print(show)
	cv2.putText(nimg,show,(10,50), font, 1,(255,255,255),2,cv2.LINE_AA)

	
	imstack = np.vstack((imstack,nimg))

	cv2.imshow("stack",imstack)
	
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break





'''
while(1):
	ret, img = cap.read()
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	
	cv2.rectangle(img,(left,top),(right,bottom),(0,0,0),1)

	img = cv2.GaussianBlur(img, (3, 3), 0)
	roi = img[top:bottom, right:left]

	
	

	s1 = cv2.resize(img,(300,300))
	s2 = cv2.resize(roi,(300,300))
	imstack = np.hstack((s1,s2))
	
	print(imstack.shape)

	

	nimg = np.zeros((300,600),np.uint8)
	print("black")
	print(nimg.shape)
	imstack = np.vstack((imstack,nimg))
	cv2.imshow("stack",imstack)
	
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break

	test_image = image.load_img('i.png', target_size = (64, 64))
	print ("Done")
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
	result = loaded_model.predict(test_image)
	#print(indices)
	print(result)
	print(result*100)
	#print('Predicted:', decode_predictions(result))
	print(np.argmax(result))

	if cv2.waitKey(1) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break

cap.release()
cv2.destroyAllWindows()


img1 = cv2.imread("i.png")
s = cv2.resize(img1,(512,512))
nimg = np.zeros((512,512,3),np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.line(nimg,(0,0),(511,511),(255,0,0),5)
cv2.putText(nimg,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
img2 = cv2.resize(nimg,(512,512))
imstack = np.hstack(s,img2)
cv2.imshow("stack",imstack)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
	
	















