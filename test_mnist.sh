#!/usr/bin/env sh
set -e

./build/tools/caffe test -gpu 0 -model examples/mnist/lenet_train_test.prototxt -weights examples/mnist/lenet_iter_10000.caffemodel -iterations 100
