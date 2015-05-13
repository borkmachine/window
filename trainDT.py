import sklearn
import scipy.io as sio
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
import threading
import multiprocessing as mp
from sklearn.externals import joblib

#mp.current_process().daemon
# threading.current_thread().name

import sklearn
#sklearn.externals.joblib.parallel.__file__

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
# save the model
joblib.dump(clf, 'randForestCollectedPics.pkl')

t = time.time()
#scores = cross_val_score(clf, feat, labels, cv=KF, n_jobs=1)
#print scores.mean()
print "Cross validation time: " + str(time.time()-t)
