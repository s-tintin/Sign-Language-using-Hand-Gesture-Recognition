import numpy as np
import cv2
import pickle


name = "A/"
i=0
l=[]

while (i<242):
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
i=0
l=[]

while (i<259):
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
i=0
l=[]

while (i<247):
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
i=0
l=[]

while (i<147):
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
i=0
l=[]

while (i<243):
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
i=0
l=[]

while (i<226):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("f_hue","w") as f:
	pickle.dump(l,f)

with open('r_f_hue', 'w') as f:
    f.write(str(l))

name = "G/"
i=0
l=[]

while (i<241):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("g_hue","w") as f:
	pickle.dump(l,f)

with open('r_g_hue', 'w') as f:
    f.write(str(l))

name = "H/"
i=0
l=[]

while (i<116):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("h_hue","w") as f:
	pickle.dump(l,f)

with open('r_h_hue', 'w') as f:
    f.write(str(l))

name = "I/"
i=0
l=[]

while (i<179):
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
i=0
l=[]

while (i<245):
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
i=0
l=[]

while (i<182):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("l_hue","w") as f:
	pickle.dump(l,f)

with open('r_l_hue', 'w') as f:
    f.write(str(l))

name = "M/"
i=0
l=[]

while (i<235):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("m_hue","w") as f:
	pickle.dump(l,f)

with open('r_m_hue', 'w') as f:
    f.write(str(l))

name = "N/"
i=0
l=[]

while (i<237):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("n_hue","w") as f:
	pickle.dump(l,f)

with open('r_n_hue', 'w') as f:
    f.write(str(l))

name = "O/"
i=0
l=[]

while (i<229):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("o_hue","w") as f:
	pickle.dump(l,f)

with open('r_o_hue', 'w') as f:
    f.write(str(l))

name = "P/"
i=0
l=[]

while (i<234):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("p_hue","w") as f:
	pickle.dump(l,f)

with open('r_p_hue', 'w') as f:
    f.write(str(l))

name = "Q/"
i=0
l=[]

while (i<216):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("q_hue","w") as f:
	pickle.dump(l,f)

with open('r_q_hue', 'w') as f:
    f.write(str(l))

name = "R/"
i=0
l=[]

while (i<211):
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
i=0
l=[]

while (i<229):
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
i=0
l=[]

while (i<216):
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
i=0
l=[]

while (i<135):
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
i=0
l=[]

while (i<122):
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
i=0
l=[]

while (i<128):
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
i=0
l=[]

while (i<202):
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
i=0
l=[]

while (i<251):
	image = name +str(i)+".png"
	img1 = cv2.imread(image,0)
	
	hm = cv2.HuMoments(cv2.moments(img1)).flatten()
	
	l.append(hm)
	i=i+1

with open("y_hue","w") as f:
	pickle.dump(l,f)

with open('r_y_hue', 'w') as f:
    f.write(str(l))





