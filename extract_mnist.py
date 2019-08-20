import sys

caffe_root = './'
sys.path.insert(0, caffe_root + 'python')

#from caffe.proto import caffe_pb2
#import google.protobuf.text_format

import caffe

net = caffe.Net('examples/mnist/lenet_train_test.prototxt', 'examples/mnist/lenet_iter_10000.caffemodel', caffe.TEST)
pic = net.blobs['data'].data
weight_conv1 = net.params['conv1'][0].data
bias_conv1 = net.params['conv1'][1].data

print(weight_conv1.shape)
print(bias_conv1.shape)



with open("weight_conv1.txt",'w') as f:
    print >>f, bias_conv1.shape[0]
    for bias in bias_conv1:
        print >>f, bias
    print >>f, weight_conv1.shape[0]
    print >>f, weight_conv1.shape[1]
    print >>f, weight_conv1.shape[2]
    print >>f, weight_conv1.shape[3]
    for chan in weight_conv1:
        for chan_in in chan:
            for row in chan_in:
                for val in row:
                    print >>f, val




weight_conv2 = net.params['conv2'][0].data
bias_conv2 = net.params['conv2'][1].data

print(weight_conv2.shape)
print(bias_conv2.shape)



with open("weight_conv2.txt",'w') as f:
    print >>f, bias_conv2.shape[0]
    for bias in bias_conv2:
        print >>f, bias
    print >>f, weight_conv2.shape[0]
    print >>f, weight_conv2.shape[1]
    print >>f, weight_conv2.shape[2]
    print >>f, weight_conv2.shape[3]
    for chan in weight_conv2:
        for chan_in in chan:
            for row in chan_in:
                for val in row:
                    print >>f, val




#ip1
weight_ip1 = net.params['ip1'][0].data
bias_ip1 = net.params['ip1'][1].data

print(weight_ip1.shape)
print(bias_ip1.shape)



with open("weight_ip1.txt",'w') as f:
    print >>f, bias_ip1.shape[0]
    for val in bias_ip1:
        print >>f, val
    print >>f, weight_ip1.shape[0]
    print >>f, weight_ip1.shape[1]
    for row in weight_ip1:
        for val in row:
            print >>f, val





#ip2
weight_ip2 = net.params['ip2'][0].data
bias_ip2 = net.params['ip2'][1].data

print(weight_ip2.shape)
print(bias_ip2.shape)



with open("weight_ip2.txt",'w') as f:
    print >>f, bias_ip2.shape[0]
    for val in bias_ip2:
        print >>f, val
    print >>f, weight_ip2.shape[0]
    print >>f, weight_ip2.shape[1]
    for row in weight_ip2:
        for val in row:
            print >>f, val



