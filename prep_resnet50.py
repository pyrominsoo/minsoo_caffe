
import sys

caffe_root = './'
sys.path.insert(0, caffe_root + 'python')

#from caffe.proto import caffe_pb2
#import google.protobuf.text_format

import caffe

net = caffe.Net('models/resnet/resnet50.prototxt', 'models/resnet/ResNet-50-model.caffemodel', caffe.TEST)
