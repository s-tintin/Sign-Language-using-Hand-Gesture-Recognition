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
import csv
import sklearn.metrics as sm
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB as nb
from sklearn.linear_model import LogisticRegression as lr
from sklearn.neural_network import MLPClassifier as mlp


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

with open("g_hue","r") as f:
	l6 = pickle.load(f)

with open("h_hue","r") as f:
	l7 = pickle.load(f)

with open("i_hue","r") as f:
	l8 = pickle.load(f)

with open("k_hue","r") as f:
	l9 = pickle.load(f)

with open("l_hue","r") as f:
	l10 = pickle.load(f)

with open("m_hue","r") as f:
	l11 = pickle.load(f)

with open("n_hue","r") as f:
	l12 = pickle.load(f)

with open("o_hue","r") as f:
	l13 = pickle.load(f)

with open("p_hue","r") as f:
	l14 = pickle.load(f)

with open("q_hue","r") as f:
	l15 = pickle.load(f)

with open("r_hue","r") as f:
	l16 = pickle.load(f)

with open("s_hue","r") as f:
	l17 = pickle.load(f)

with open("t_hue","r") as f:
	l18 = pickle.load(f)

with open("u_hue","r") as f:
	l19 = pickle.load(f)

with open("v_hue","r") as f:
	l20 = pickle.load(f)

with open("w_hue","r") as f:
	l21 = pickle.load(f)

with open("x_hue","r") as f:
	l22 = pickle.load(f)

with open("y_hue","r") as f:
	l23 = pickle.load(f)

X = np.concatenate((l0, l1), axis=0)
#print("AB"+str(len(X)))
X = np.concatenate((X, l2), axis=0)
#print("C"+str(len(X)))
X = np.concatenate((X, l3), axis=0)
#print("D"+str(len(X)))
X = np.concatenate((X, l4), axis=0)
#print("E"+str(len(X)))
X = np.concatenate((X, l5), axis=0)
#print("F"+str(len(X)))
X = np.concatenate((X, l6), axis=0)
#print("I"+str(len(X)))
X = np.concatenate((X, l7), axis=0)
#print("K"+str(len(X)))
X = np.concatenate((X, l8), axis=0)
#print("L"+str(len(X)))
X = np.concatenate((X, l9), axis=0)
#print("N"+str(len(X)))
X = np.concatenate((X, l10), axis=0)
#print("R"+str(len(X)))
X = np.concatenate((X, l11), axis=0)
#print("S"+str(len(X)))
X = np.concatenate((X, l12), axis=0)
#print("T"+str(len(X)))
X = np.concatenate((X, l13), axis=0)
#print("U"+str(len(X)))
X = np.concatenate((X, l14), axis=0)
#print("V"+str(len(X)))
X = np.concatenate((X, l15), axis=0)
#print("W"+str(len(X)))
X = np.concatenate((X, l16), axis=0)
#print("X"+str(len(X)))
X = np.concatenate((X, l17), axis=0)
#print("Y"+str(len(X)))
X = np.concatenate((X, l18), axis=0)
X = np.concatenate((X, l19), axis=0)
X = np.concatenate((X, l20), axis=0)
X = np.concatenate((X, l21), axis=0)
X = np.concatenate((X, l22), axis=0)
X = np.concatenate((X, l23), axis=0)


print("X",len(X))



scalar = StandardScaler()
x=scalar.fit(X)

X= scalar.transform(X)


a = np.zeros((1,242))

b= np.ones((1,259))

c = np.full((1,247), 2)

d = np.full((1,147), 3)

e = np.full((1,243), 4)

f = np.full((1,226), 5)

g = np.full((1,241), 6)

h = np.full((1,116), 7)

i = np.full((1,179), 8)

k = np.full((1,245), 9)

l = np.full((1,182), 10)

m = np.full((1,235), 11)

n = np.full((1,237), 12)

o = np.full((1,229), 13)

p = np.full((1,234), 14)

q = np.full((1,216), 15)

r = np.full((1,211), 16)

s = np.full((1,229), 17)

t = np.full((1,216), 18)

u = np.full((1,135), 19)

v = np.full((1,122), 20)

w = np.full((1,128), 21)

x = np.full((1,202), 22)

y = np.full((1,251), 23)



Y = np.append(a, b)
Y = np.append(Y, c)
Y = np.append(Y, d)
Y = np.append(Y, e)
Y = np.append(Y, f)
Y = np.append(Y, g)
Y = np.append(Y, h)
Y = np.append(Y, i)
Y = np.append(Y, k)
Y = np.append(Y, l)
Y = np.append(Y, m)
Y = np.append(Y, n)
Y = np.append(Y, o)
Y = np.append(Y, p)
Y = np.append(Y, q)
Y = np.append(Y, r)
Y = np.append(Y, s)
Y = np.append(Y, t)
Y = np.append(Y, u)
Y = np.append(Y, v)
Y = np.append(Y, w)
Y = np.append(Y, x)
Y = np.append(Y, y)


print("Y",len(Y))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)


def calc_accuracy(method,label_test,pred):
    print("accuracy score for ",method,sm.accuracy_score(label_test,pred))
    print("precision_score for ",method,sm.precision_score(label_test,pred,average='micro'))
    print("f1 score for ",method,sm.f1_score(label_test,pred,average='micro'))
    print("recall score for ",method,sm.recall_score(label_test,pred,average='micro'))

def predict_svm(X_train, X_test, y_train, y_test):
    svc=SVC(kernel='linear') 
    print("svm started")
    svc.fit(X_train,y_train)
    pickle.dump(svc,open("svc_hu_model.pkl",'wb'))
    y_pred=svc.predict(X_test)
    calc_accuracy("SVM",y_test,y_pred)
    np.savetxt('submission_hu_svm.csv', np.c_[range(1,len(y_test)+1),y_pred,y_test], delimiter=',', header = 'ImageId,Label,TrueLabel', comments = '', fmt='%d')
    

def predict_lr(X_train, X_test, y_train, y_test):
    clf = lr()
    print("lr started")
    clf.fit(X_train,y_train)
    pickle.dump(clf,open("lr_hu_model.pkl",'wb'))
    y_pred=clf.predict(X_test)
    calc_accuracy("Logistic regression",y_test,y_pred)
    np.savetxt('submission_hu_lr.csv', np.c_[range(1,len(y_test)+1),y_pred,y_test], delimiter=',', header = 'ImageId,Label,TrueLabel', comments = '', fmt='%d')
    


def predict_nb(X_train, X_test, y_train, y_test):
    clf = nb()
    print("nb started")
    clf.fit(X_train,y_train)
    pickle.dump(clf,open("nb_hu_model.pkl",'wb'))
    y_pred=clf.predict(X_test)
    calc_accuracy("Naive Bayes",y_test,y_pred)
    np.savetxt('submission_hu_nb.csv', np.c_[range(1,len(y_test)+1),y_pred,y_test], delimiter=',', header = 'ImageId,Label,TrueLabel', comments = '', fmt='%d')
   
    
def predict_mlp(X_train, X_test, y_train, y_test):
    clf=mlp()
    print("mlp started")
    clf.fit(X_train,y_train)
    pickle.dump(clf,open("mlp_hu_model.pkl",'wb'))
    y_pred=clf.predict(X_test)
    calc_accuracy("MLP classifier",y_test,y_pred)
    np.savetxt('submission_hu_mlp.csv', np.c_[range(1,len(y_test)+1),y_pred,y_test], delimiter=',', header = 'ImageId,Label,TrueLabel', comments = '', fmt='%d')


#using classification methods

predict_svm(X_train, X_test,y_train, y_test)

predict_lr(X_train, X_test,y_train, y_test)
predict_nb(X_train, X_test,y_train, y_test)
predict_mlp(X_train, X_test,y_train, y_test)






