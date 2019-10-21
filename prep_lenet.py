
import sys

caffe_root = './'
sys.path.insert(0, caffe_root + 'python')

#from caffe.proto import caffe_pb2
#import google.protobuf.text_format

import caffe

net = caffe.Net('examples/mnist/lenet_train_test.prototxt', 'examples/mnist/lenet_iter_10000.caffemodel', caffe.TEST)
