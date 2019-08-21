import sys

caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')

from caffe.proto import caffe_pb2
import google.protobuf.text_format
net = caffe_pb2.NetParameter()
f = open('resnet50.prototxt', 'r')
net = google.protobuf.text_format.Merge(str(f.read()), net)
f.close()

print(len(net.layer))

for layer in net.layer:
    if layer.name.startswith('bn'):
        layerNames = [l.name for l in net.layer]
        idx = layerNames.index(layer.name)
        del net.layer[idx]

print(len(net.layer))

for layer in net.layer:
    if layer.name.startswith('scale'):
        layerNames = [l.name for l in net.layer]
        idx = layerNames.index(layer.name)
        del net.layer[idx]

print(len(net.layer))

with open('resnet50_nobatchnorm.prototxt', 'w') as f:
    f.write(str(net))

