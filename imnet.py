# This file runs a candidate image through the CNN, and then passes the feature vector from fc7 to a pre-trained random forest. 
# Currently, this will create a matrix with [index, {0,1}] where 0 means not a window, 1 means window, according to the forest. 

# Important things in this version:
## imprefix: this is the name of the region proposal candidate images, such as “/home/chiller/cropped/crop_42.jpg” 
### the current structure of the code assumes the file names with number.jpg but this can easily be changed. 

## data is where the matrix of ID’s in variable “valid_ids” is stored, this is a list of valid indices
### (as determined by the normal map filtering or other processing) that we wish to classify. 

## clf is the classifier, and is loaded from the specificed path/pkl file. I don’t think this has to be a random forest, but that is what we used. 

## As written, this program will create n copies of itself where n is the number of processor threads available.
### On Quebec, this is 32. The only reason that we are running caffe in CPU mode is that GPU mode wasn’t working 
### right off the bat and we didn’t want to spend time fixing it. Running this on the GPU should make it much much faster. 

import numpy as np

from joblib import Parallel, delayed
import multiprocessing
import sys
import caffe
import csv

import sklearn
import scipy.io as sio
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import scipy.io as sio

caffe.set_mode_cpu()

ROOT = '/home/chiller/'
MODEL_FILE = ROOT + 'caffe/models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = ROOT + '/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'


net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(ROOT + '/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

# Random Forest classifier
clf = joblib.load("/home/chiller/randForestCollectedPics2.pkl")


def processInput(i,imprefix):
	#IMAGE_FILE = "/home/chiller/400coryIms_masked/Camera_110732781_Image000001.jpg"	
	#IMAGE_FILE = ROOT + "cropped/"+ "crop_"+str(i)+".jpg"
	IMAGE_FILE = imprefix + str(i) + ".jpg"	
	input_image = caffe.io.load_image(IMAGE_FILE)		
	prediction = net.predict([input_image])
	featVec = net.blobs['fc7'].data[0]
	windowPrediction = clf.predict(featVec)
	print int(windowPrediction[0])
	return [i, int(windowPrediction[0])] #+ list(int(windowPrediction[0]))
	#return [i] + list(net.blobs['fc7'].data[0])


def main(inputs,imprefix,n=0):
	#inputs = range(1,n+1)

	num_cores = multiprocessing.cpu_count()
	results = Parallel(n_jobs=num_cores)(delayed(processInput)(i,imprefix) for i in inputs)
	results = np.matrix([res[0] for res in results if res[1]==1])
	return results

if __name__ == '__main__':
	imprefix = "/home/chiller/cropped/crop_"
	data = sio.loadmat('IDProps.mat')
	if data['valid_ids'].shape[0] == 0:
		res = []
	else:
		indicies = data['valid_ids'][0]

		#print indicies
		#print data.keys()
		#processInput(34)
		#n = int(sys.argv[1])
		res = main(list(indicies), imprefix)
	sio.savemat('outputVec.mat', {'idx':res})
	print res
	with open('outputTest.csv','wb') as f:
		writer = csv.writer(f)
		writer.writerows(res)

