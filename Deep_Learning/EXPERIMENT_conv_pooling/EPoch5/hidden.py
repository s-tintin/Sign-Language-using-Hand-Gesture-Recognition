import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras import backend as K
from keras.models import Model
import cv2
import numpy as np

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
h=loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model = loaded_model

w0 = model.layers[0].get_weights()
k=0
text=""
for i in w0:
	text=text+str(k)+"\n"
	text=text+str(i)+"\n"
	k=k+1
f = open("hiddenoutput.txt","w") 
f.write(text)
f.close 
