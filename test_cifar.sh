#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe test -gpu 0 -model examples/cifar10/cifar10_full_train_test.prototxt -weights examples/cifar10/cifar10_full_iter_70000.caffemodel.h5 -iterations 100
