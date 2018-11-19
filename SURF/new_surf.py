import numpy as np
import cv2
import os
import csv
import sklearn.metrics as sm
from surf_image_processing import func
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
import random
import warnings
import pickle
from sklearn.naive_bayes import GaussianNB as nb
from sklearn.linear_model import LogisticRegression as lr
import numpy as np
import sklearn.metrics as sm
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier as mlp

#initialise
path="train"
label=0
img_descs=[]
y=[]

#utility functions
def perform_data_split(X, y, training_idxs, test_idxs, val_idxs):
    """
    Split X and y into train/test/val sets
    Parameters:
    -----------
    X : eg, use img_bow_hist
    y : corresponding labels for X
    training_idxs : list/array of integers used as indicies for training rows
    test_idxs : same
    val_idxs : same
    Returns:
    --------
    X_train, X_test, X_val, y_train, y_test, y_val
    """
    X_train = X[training_idxs]
    X_test = X[test_idxs]
    #X_val = X[val_idxs]

    y_train = y[training_idxs]
    y_test = y[test_idxs]
    #y_val = y[val_idxs]

    return X_train, X_test, y_train, y_test, 

def train_test_val_split_idxs(total_rows, percent_test, percent_val):
    """
    Get indexes for training, test, and validation rows, given a total number of rows.
    Assumes indexes are sequential integers starting at 0: eg [0,1,2,3,...N]
    Returns:
    --------
    training_idxs, test_idxs, val_idxs
        Both lists of integers
    """
    if percent_test + percent_val >= 1.0:
        raise ValueError('percent_test and percent_val must sum to less than 1.0')

    row_range = range(total_rows)

    no_test_rows = int(total_rows*(percent_test))
    if (no_test_rows > 0):
    	test_idxs = np.random.choice(row_range, size=no_test_rows, replace=False)
    else:
    	test_idxs = np.array([])
    # remove test indexes
    row_range = [idx for idx in row_range if idx not in test_idxs]

    no_val_rows = int(total_rows*(percent_val))
    if(no_val_rows > 0):
    	val_idxs = np.random.choice(row_range, size=no_val_rows, replace=False)
    else:
	val_idxs = np.array([])
    # remove validation indexes
    training_idxs = [idx for idx in row_range if idx not in val_idxs]
    print(training_idxs)

    print('Train-test-val split: %i training rows, %i test rows, %i validation rows' % (len(training_idxs), len(test_idxs), len(val_idxs)))

    return training_idxs, test_idxs, val_idxs

def cluster_features(img_descs, training_idxs, cluster_model):
    """
    Cluster the training features using the cluster_model
    and convert each set of descriptors in img_descs
    to a Visual Bag of Words histogram.
    Parameters:
    -----------
    X : list of lists of SIFT descriptors (img_descs)
    training_idxs : array/list of integers
        Indicies for the training rows in img_descs
    cluster_model : clustering model (eg KMeans from scikit-learn)
        The model used to cluster the SIFT features
    Returns:
    --------
    X, cluster_model :
        X has K feature columns, each column corresponding to a visual word
        cluster_model has been fit to the training set
    """
    n_clusters = cluster_model.n_clusters

    # # Generate the SIFT descriptor features
    # img_descs = gen_sift_features(labeled_img_paths)
    #
    # # Generate indexes of training rows
    # total_rows = len(img_descs)
    # training_idxs, test_idxs, val_idxs = train_test_val_split_idxs(total_rows, percent_test, percent_val)

    # Concatenate all descriptors in the training set together
    training_descs = [img_descs[i] for i in training_idxs]
    all_train_descriptors = [desc for desc_list in training_descs for desc in desc_list]
    all_train_descriptors = np.array(all_train_descriptors)


    print ('%i descriptors before clustering' % all_train_descriptors.shape[0])

    # Cluster descriptors to get codebook
    print ('Using clustering model %s...' % repr(cluster_model))
    print ('Clustering on training set to get codebook of %i words' % n_clusters)

    # train kmeans or other cluster model on those descriptors selected above
    cluster_model.fit(all_train_descriptors)
    print ('done clustering. Using clustering model to generate BoW histograms for each image.')
    pickle.dump(cluster_model,open("cluster_model.pkl",'wb'))

    # compute set of cluster-reduced words for each image
    img_clustered_words = [cluster_model.predict(raw_words) for raw_words in img_descs]

    # finally make a histogram of clustered word counts for each image. These are the final features.
    img_bow_hist = np.array(
        [np.bincount(clustered_words, minlength=n_clusters) for clustered_words in img_clustered_words])

    X = img_bow_hist
    print ('done generating BoW histograms.')
    print(X)

    return X, cluster_model

def calc_accuracy(method,label_test,pred):
    print("accuracy score for ",method,sm.accuracy_score(label_test,pred))
    print("precision_score for ",method,sm.precision_score(label_test,pred,average='micro'))
    print("f1 score for ",method,sm.f1_score(label_test,pred,average='micro'))
    print("recall score for ",method,sm.recall_score(label_test,pred,average='micro'))

def predict_svm(X_train, X_test, y_train, y_test):
    svc=SVC(kernel='linear') 
    print("svm started")
    svc.fit(X_train,y_train)
    pickle.dump(svc,open("svc_model.pkl",'wb'))
    y_pred=svc.predict(X_test)
    calc_accuracy("SVM",y_test,y_pred)
    np.savetxt('submission_surf_svm.csv', np.c_[range(1,len(y_test)+1),y_pred,y_test], delimiter=',', header = 'ImageId,Label,TrueLabel', comments = '', fmt='%d')
    #mat = confusion_matrix(y_test, y_pred, labels=['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y'])
    #print("Confusion matrix of SVM")
    #print(mat)
    

def predict_lr(X_train, X_test, y_train, y_test):
    clf = lr()
    print("lr started")
    clf.fit(X_train,y_train)
    pickle.dump(clf,open("lr_model.pkl",'wb'))
    y_pred=clf.predict(X_test)
    calc_accuracy("Logistic regression",y_test,y_pred)
    np.savetxt('submission_surf_lr.csv', np.c_[range(1,len(y_test)+1),y_pred,y_test], delimiter=',', header = 'ImageId,Label,TrueLabel', comments = '', fmt='%d')
    #mat = confusion_matrix(y_test, y_pred, labels=['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y'])
    #print("Confusion matrix of Logistic Regression")
    #print(mat)
    


def predict_nb(X_train, X_test, y_train, y_test):
    clf = nb()
    print("nb started")
    clf.fit(X_train,y_train)
    pickle.dump(clf,open("nb_model.pkl",'wb'))
    y_pred=clf.predict(X_test)
    calc_accuracy("Naive Bayes",y_test,y_pred)
    np.savetxt('submission_surf_nb.csv', np.c_[range(1,len(y_test)+1),y_pred,y_test], delimiter=',', header = 'ImageId,Label,TrueLabel', comments = '', fmt='%d')
    #mat = confusion_matrix(y_test, y_pred, labels=['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y'])
    #print("Confusion matrix of Naive Bayes")
    #print(mat)
    
def predict_mlp(X_train, X_test, y_train, y_test):
    clf=mlp()
    print("mlp started")
    clf.fit(X_train,y_train)
    pickle.dump(clf,open("mlp_model.pkl",'wb'))
    y_pred=clf.predict(X_test)
    calc_accuracy("MLP classifier",y_test,y_pred)
    np.savetxt('submission_surf_mlp.csv', np.c_[range(1,len(y_test)+1),y_pred,y_test], delimiter=',', header = 'ImageId,Label,TrueLabel', comments = '', fmt='%d')

#print("Yesssss")
#creating desc for each file with label
for (dirpath,dirnames,filenames) in os.walk(path):
    #print("Yesssss inside 1")
    for dirname in dirnames:
        print(dirname)
        for(direcpath,direcnames,files) in os.walk(path+"/"+dirname):
	    #print("Yesssss inside 2")
            for file in files:
                actual_path=path+"/"+dirname+"/"+file
		#print("Yesssss")
                #print("actual path",actual_path)
                   
    		frame = cv2.imread(actual_path)
	    	frame = cv2.resize(frame,(128,128))
	    	converted2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert from RGB to HSV
		#cv2.imshow("original",converted2)

		lowerBoundary = np.array([0,40,30],dtype="uint8")
		upperBoundary = np.array([43,255,254],dtype="uint8")
		skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)
		skinMask = cv2.addWeighted(skinMask,0.5,skinMask,0.5,0.0)
		#cv2.imshow("masked",skinMask)
    
    		skinMask = cv2.medianBlur(skinMask, 5)
    
    		skin = cv2.bitwise_and(converted2, converted2, mask = skinMask)
    		#frame = cv2.addWeighted(frame,1.5,skin,-0.5,0)
    		#skin = cv2.bitwise_and(frame, frame, mask = skinMask)

    		#skinGray=cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    
    		#cv2.imshow("masked2",skin)
    		img2 = cv2.Canny(skin,60,60)
    		#cv2.imshow("edge detection",img2)
    
    		''' 
    		hog = cv2.HOGDescriptor()
   		h = hog.compute(img2)
    		print(len(h))
    
    		'''
    		surf = cv2.xfeatures2d.SURF_create()
    		#surf.extended=True
    		img2 = cv2.resize(img2,(256,256))
    		kp, des = surf.detectAndCompute(img2,None)
    		#print(len(des))
    		img2 = cv2.drawKeypoints(img2,kp,None,(0,0,255),4)
    		#plt.imshow(img2),plt.show()
    
    		#cv2.waitKey(0)
    		#cv2.destroyAllWindows()
    		print(len(des))
    
                img_descs.append(des)
                y.append(label)
		#print(label)
        label=label+1

#finding indexes of test train and validate
y=np.array(y)
training_idxs, test_idxs, val_idxs = train_test_val_split_idxs(len(img_descs), 0.4, 0.0)

#creating histogram using kmeans minibatch cluster model
X, cluster_model = cluster_features(img_descs, training_idxs, MiniBatchKMeans(n_clusters=150))

#splitting data into test, train, validate using the indexes
X_train, X_test, y_train, y_test = perform_data_split(X, y, training_idxs, test_idxs, val_idxs)


#using classification methods

predict_svm(X_train, X_test,y_train, y_test)

predict_lr(X_train, X_test,y_train, y_test)
predict_nb(X_train, X_test,y_train, y_test)
predict_mlp(X_train, X_test,y_train, y_test)


