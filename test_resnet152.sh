#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe test -model models/resnet/resnet152.prototxt -weights models/resnet/ResNet-152-model.caffemodel -iterations 100
