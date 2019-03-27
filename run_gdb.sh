#!/usr/bin/env sh
set -e

TOOLS=./build/tools

gdb --args $TOOLS/caffe test -model models/resnet/resnet50.prototxt -weights models/resnet/ResNet-50-model.caffemodel -iterations 1
