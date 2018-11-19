import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB


with open("a_hue","r") as f:
	l0 = pickle.load(f)

with open("b_hue","r") as f:
	l1 = pickle.load(f)

with open("c_hue","r") as f:
	l2 = pickle.load(f)

with open("d_hue","r") as f:
	l3 = pickle.load(f)

with open("e_hue","r") as f:
	l4 = pickle.load(f)

with open("f_hue","r") as f:
	l5 = pickle.load(f)

with open("i_hue","r") as f:
	l6 = pickle.load(f)

with open("k_hue","r") as f:
	l7 = pickle.load(f)

with open("l_hue","r") as f:
	l8 = pickle.load(f)

with open("n_hue","r") as f:
	l9 = pickle.load(f)

with open("r_hue","r") as f:
	l10 = pickle.load(f)

with open("s_hue","r") as f:
	l11 = pickle.load(f)

with open("t_hue","r") as f:
	l12 = pickle.load(f)

with open("u_hue","r") as f:
	l13 = pickle.load(f)

with open("v_hue","r") as f:
	l14 = pickle.load(f)

with open("w_hue","r") as f:
	l15 = pickle.load(f)

with open("x_hue","r") as f:
	l16 = pickle.load(f)

with open("y_hue","r") as f:
	l17 = pickle.load(f)

X = np.concatenate((l0, l1), axis=0)
X = np.concatenate((X, l2), axis=0)
X = np.concatenate((X, l3), axis=0)
X = np.concatenate((X, l4), axis=0)
X = np.concatenate((X, l5), axis=0)
X = np.concatenate((X, l6), axis=0)
X = np.concatenate((X, l7), axis=0)
X = np.concatenate((X, l8), axis=0)
X = np.concatenate((X, l9), axis=0)
X = np.concatenate((X, l10), axis=0)
X = np.concatenate((X, l11), axis=0)
X = np.concatenate((X, l12), axis=0)
X = np.concatenate((X, l13), axis=0)
X = np.concatenate((X, l14), axis=0)
X = np.concatenate((X, l15), axis=0)
X = np.concatenate((X, l16), axis=0)
X = np.concatenate((X, l17), axis=0)


print(len(X))



scalar = StandardScaler()
x=scalar.fit(X)

X= scalar.transform(X)


a = np.zeros((1,3000))

b= np.ones((1,3000))

c = np.full((1,3000), 2)

d = np.full((1,3000), 3)

e = np.full((1,3000), 4)

f = np.full((1,3000), 5)

i = np.full((1,3000), 6)

k = np.full((1,3000), 7)

l = np.full((1,3000), 8)

n= np.full((1,3000), 9)

r = np.full((1,3000), 10)

s = np.full((1,3000), 11)

t = np.full((1,3000), 12)

u = np.full((1,3000), 13)

v = np.full((1,3000), 14)

w = np.full((1,3000), 15)

x = np.full((1,3000), 16)

y = np.full((1,3000), 17)



Y = np.append(a, b)
Y = np.append(Y, c)
Y = np.append(Y, d)
Y = np.append(Y, e)
Y = np.append(Y, f)
Y = np.append(Y, i)
Y = np.append(Y, k)
Y = np.append(Y, l)
Y = np.append(Y, n)
Y = np.append(Y, r)
Y = np.append(Y, s)
Y = np.append(Y, t)
Y = np.append(Y, u)
Y = np.append(Y, v)
Y = np.append(Y, w)
Y = np.append(Y, x)
Y = np.append(Y, y)


print(len(Y))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

gnb = GaussianNB()

model = gnb.fit(X_train, y_train)

tuple_objects = (scalar,model)
pickle.dump(tuple_objects,open("tuple.pkl",'wb'))




e = X_test[0]
y_predict = model.predict(X_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy=")
print(acc)

print("printing l")

mat = confusion_matrix(y_test, y_predict, labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
print(mat)

print(model.predict([X_test[0]]))
print(y_test[0])






