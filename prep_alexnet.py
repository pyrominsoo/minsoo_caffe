
import sys

caffe_root = './'
sys.path.insert(0, caffe_root + 'python')

#from caffe.proto import caffe_pb2
#import google.protobuf.text_format

import caffe

net = caffe.Net('models/bvlc_alexnet/train_val.prototxt', 'models/bvlc_alexnet/bvlc_alexnet.caffemodel', caffe.TEST)
