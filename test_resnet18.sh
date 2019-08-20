#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe test -model models/resnet18/train.prototxt -weights models/resnet18/resnet18_iter_1000000.caffemodel -iterations 100
