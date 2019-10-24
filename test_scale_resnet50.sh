#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe test -gpu 0 -model models/resnet_scale/resnet50_scale.prototxt -weights models/resnet_scale/resnet50_scale.caffemodel -iterations 143
