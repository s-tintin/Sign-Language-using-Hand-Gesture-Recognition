import numpy as np
import cv2
import pickle


name = "A/"
i=1
l=[]

while (i<3001):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("a_hue","w") as f:
	pickle.dump(l,f)

with open('r_a_hue', 'w') as f:
    f.write(str(l))

name = "B/"
i=1
l=[]

while (i<3001):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("b_hue","w") as f:
	pickle.dump(l,f)

with open('r_b_hue', 'w') as f:
    f.write(str(l))


name = "C/"
i=1
l=[]

while (i<3001):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("c_hue","w") as f:
	pickle.dump(l,f)

with open('r_c_hue', 'w') as f:
    f.write(str(l))


name = "D/"
i=1
l=[]

while (i<3001):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("d_hue","w") as f:
	pickle.dump(l,f)

with open('r_d_hue', 'w') as f:
    f.write(str(l))


name = "E/"
i=1
l=[]

while (i<3001):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("e_hue","w") as f:
	pickle.dump(l,f)

with open('r_e_hue', 'w') as f:
    f.write(str(l))


name = "F/"
i=1
l=[]

while (i<3001):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("f_hue","w") as f:
	pickle.dump(l,f)

with open('r_f_hue', 'w') as f:
    f.write(str(l))


name = "I/"
i=1
l=[]

while (i<3001):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("i_hue","w") as f:
	pickle.dump(l,f)

with open('r_i_hue', 'w') as f:
    f.write(str(l))


name = "K/"
i=1
l=[]

while (i<3001):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("k_hue","w") as f:
	pickle.dump(l,f)

with open('r_k_hue', 'w') as f:
    f.write(str(l))


name = "L/"
i=1
l=[]

while (i<3001):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("l_hue","w") as f:
	pickle.dump(l,f)

with open('r_l_hue', 'w') as f:
    f.write(str(l))


name = "N/"
i=1
l=[]

while (i<3001):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("n_hue","w") as f:
	pickle.dump(l,f)

with open('r_n_hue', 'w') as f:
    f.write(str(l))


name = "R/"
i=1
l=[]

while (i<3001):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("r_hue","w") as f:
	pickle.dump(l,f)

with open('r_r_hue', 'w') as f:
    f.write(str(l))


name = "S/"
i=1
l=[]

while (i<3001):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("s_hue","w") as f:
	pickle.dump(l,f)

with open('r_s_hue', 'w') as f:
    f.write(str(l))


name = "T/"
i=1
l=[]

while (i<3001):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("t_hue","w") as f:
	pickle.dump(l,f)

with open('r_t_hue', 'w') as f:
    f.write(str(l))


name = "U/"
i=1
l=[]

while (i<3001):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("u_hue","w") as f:
	pickle.dump(l,f)

with open('r_u_hue', 'w') as f:
    f.write(str(l))


name = "V/"
i=1
l=[]

while (i<3001):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("v_hue","w") as f:
	pickle.dump(l,f)

with open('r_v_hue', 'w') as f:
    f.write(str(l))


name = "W/"
i=1
l=[]

while (i<3001):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("w_hue","w") as f:
	pickle.dump(l,f)

with open('r_w_hue', 'w') as f:
    f.write(str(l))


name = "X/"
i=1
l=[]

while (i<3001):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("x_hue","w") as f:
	pickle.dump(l,f)

with open('r_x_hue', 'w') as f:
    f.write(str(l))


name = "Y/"
i=1
l=[]

while (i<3001):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("y_hue","w") as f:
	pickle.dump(l,f)

with open('r_y_hue', 'w') as f:
    f.write(str(l))





