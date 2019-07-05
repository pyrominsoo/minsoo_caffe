#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe test -model models/resnet18_no_batnorm/train.prototxt -weights models/resnet18_no_batnorm/no_batnorm_iter_200000.caffemodel -iterations 1
