#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe test -gpu 0 -model models/resnet/resnet101.prototxt -weights models/resnet/ResNet-101-model.caffemodel -iterations 2000
