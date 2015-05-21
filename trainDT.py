#This scripts reads from data.mat, one variable contains the features, another contains labels. 
#Currently this uses a RandomForestClassifier, but this could be changed to whatever you want. 
#For sklearn reference: http://scikit-learn.org/stable/modules/classes.html#reference

import sklearn
import scipy.io as sio
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
import threading
import multiprocessing as mp



data = sio.loadmat('data.mat')
feat = data['feat']
labels = data['labels']
labels = labels[0,:]

m,n = feat.shape


KF = KFold(n=labels.size, n_folds=3, shuffle=True)

t = time.time()
clf = RandomForestClassifier(n_estimators=100, max_depth=512, min_samples_split=1, random_state=0, n_jobs=-1)
clf.fit(feat, labels)
print "Classifier build time: " + str(time.time()-t)

# save the model.
joblib.dump(clf, 'randForestCollectedPics.pkl')

# cross validate the model.
t = time.time()
scores = cross_val_score(clf, feat, labels, cv=KF, n_jobs=1)
print scores.mean()
print "Cross validation time: " + str(time.time()-t)
