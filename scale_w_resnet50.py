
import sys

caffe_root = './'
sys.path.insert(0, caffe_root + 'python')

#from caffe.proto import caffe_pb2
#import google.protobuf.text_format

import caffe

net = caffe.Net('models/resnet/resnet50.prototxt', 'models/resnet/ResNet-50-model.caffemodel', caffe.TEST)
new_net = caffe.Net('models/resnet_scale/resnet50_scale.prototxt', 'models/resnet/ResNet-50-model.caffemodel', caffe.TEST)

for idx in range(len(net.layers)):
  layer = net.layers[idx]
  if (layer.type == 'Convolution'):
    try:
      weights = layer.blobs[0].data
      new_net.layers[idx].blobs[0].data[...] = weights * 4
      if (len(layer.blobs) == 2):
        biases = layer.blobs[1].data
        new_net.layers[idx].blobs[1].data[...] = biases * 4
    except:
      print(idx)
      print(list(net.layers)[idx])
  elif (layer.type == 'BatchNorm'):
    try:
      means = layer.blobs[0].data
      varis = layer.blobs[1].data
      scale_factor = layer.blobs[2].data
      new_net.layers[idx].blobs[0].data[...] = means * 4
      new_net.layers[idx].blobs[1].data[...] = varis * 16
      new_net.layers[idx].blobs[2].data[...] = scale_factor
    except:
      print(idx)
      print(list(net.layers)[idx])
new_net.save('models/resnet_scale/resnet50_scale.caffemodel')

