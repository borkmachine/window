import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import sys
import caffe
import csv

caffe.set_mode_cpu()
ROOT = '/home/chiller/'
MODEL_FILE = ROOT + 'caffe/models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = ROOT + '/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'


net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(ROOT + '/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))
f = open(ROOT + "/synset_words.txt")
lines = f.readlines()
lines = [line.strip()[10:] for line in lines]
category = {}
for k,v in enumerate(lines):
	category[k]=v
def processInput(i):
	#for i in range(1,1538):
	#print i
	#IMAGE_FILE = ROOT + "window/"+ "SelectiveSearchCodeIJCV/blobs/blob_"+str(i)+".jpg"
	IMAGE_FILE = ROOT + "cropped/"+ "crop_"+str(i)+".jpg"
	input_image = caffe.io.load_image(IMAGE_FILE)		
	prediction = net.predict([input_image])
	#print net.blobs['fc7'].data[0].shape

	return [i] + list(net.blobs['fc7'].data[0])
	#print str(i) + ',' + ','.join(str(x) for x in list(net.blobs['fc7'].data[0]))+","
	#print prediction[0].argmax()
	#top5 = set(prediction[0].argsort()[-5:])
	#windowNums = set([904, 905, 580, 799]) # Magic numbers for window/window like things
	#if windowNums & top5:
	#	print str(i) + ","
	#	#return i

def main(n):
	inputs = range(1,n+1)
	num_cores = multiprocessing.cpu_count()
	#print num_cores
	results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)
	return results

if __name__ == '__main__':
	n = int(sys.argv[1])
	res = main(n)
	#print res
	#print len(results)
	#res = [v for v in res if v is not None]
	#print res
	with open('output.csv','wb') as f:
		writer = csv.writer(f)
		writer.writerows(res)
	#sys.stdout.write(str(res))
