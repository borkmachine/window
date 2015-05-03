import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing

import caffe

caffe.set_mode_cpu()

MODEL_FILE = '../caffe/models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = '../caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'


net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load('../caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))
f = open("../synset_words.txt")
lines = f.readlines()
lines = [line.strip()[10:] for line in lines]
category = {}
for k,v in enumerate(lines):
	category[k]=v

inputs = range(1,1538)
def processInput(i):
#for i in range(1,1538):
	#print i
	IMAGE_FILE = "../blobs/blob_"+str(i)+".jpg"
	
	input_image = caffe.io.load_image(IMAGE_FILE)		
	prediction = net.predict([input_image])
	#print prediction[0].argmax()
	top5 = set(prediction[0].argsort()[-5:])
	windowNums = set([904, 905, 580, 799]) # Magic numbers for window/window like things
	if windowNums & top5:
		print i
		return i
num_cores = multiprocessing.cpu_count()
print num_cores
results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)
print results

