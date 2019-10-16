#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe test -gpu 0 -model models/resnet/resnet50.prototxt -weights models/resnet/ResNet-50-model.caffemodel -iterations 143
