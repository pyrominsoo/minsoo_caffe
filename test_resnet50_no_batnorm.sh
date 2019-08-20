#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe test -model models/resnet50_no_batnorm/train.prototxt -weights models/resnet50_no_batnorm/no_batnorm_iter_160000.caffemodel -iterations 1
