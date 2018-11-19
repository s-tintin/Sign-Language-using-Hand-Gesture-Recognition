import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


sub = pd.read_csv("submission_hu_lr.csv")
y_pred = np.array(sub.pop('Label'))
y_test = np.array(sub.pop('TrueLabel'))
print("Confusion matrix of Logistic Regression")
mat = confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
print(mat)

sub = pd.read_csv("submission_hu_nb.csv")
y_pred = np.array(sub.pop('Label'))
y_test = np.array(sub.pop('TrueLabel'))
print("Confusion matrix of Naive Bayesian")
mat = confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
print(mat)

sub = pd.read_csv("submission_hu_svm.csv")
y_pred = np.array(sub.pop('Label'))
y_test = np.array(sub.pop('TrueLabel'))
print("Confusion matrix of SVM")
mat = confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
print(mat)

sub = pd.read_csv("submission_hu_mlp.csv")
y_pred = np.array(sub.pop('Label'))
y_test = np.array(sub.pop('TrueLabel'))
print("Confusion matrix of MLP")
mat = confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
print(mat)
