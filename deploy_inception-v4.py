import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import sys
import numpy as np
caffe_root = './'
sys.path.insert(0, caffe_root + 'python')

import caffe


model_path = caffe_root + "models/mmdnn_inception-v4/original.prototxt"
weights_path = caffe_root + "models/mmdnn_inception-v4/inception-v4.caffemodel"

caffe.set_mode_cpu()
#caffe.set_device(0)

net = caffe.Net(model_path, weights_path, caffe.TEST)


#mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
#mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
#print 'mean-subtracted values:', zip('BGR', mu)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
#transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
#transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

image = caffe.io.load_image(caffe_root+'images/ILSVRC2012_val_00000002.JPEG')
transformed_image = transformer.preprocess('data', image)

#plt.imshow(image)

net.blobs['data'].data[...] = transformed_image

output = net.forward()
output_prob = output['prob'][0]
output_prob = np.squeeze(output_prob)

print('predicted class is:', output_prob.argmax())
print('probability:', output_prob.max())

labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(labels_file, str, delimiter='\t')
print('output label:', labels[output_prob.argmax()])

top_inds = np.argsort(output_prob)[::-1][:5]

print 'probabilities and labels:' 
print zip(output_prob[top_inds], labels[top_inds])
