#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe test -gpu 0 -model models/resnet18_no_batnorm/train.prototxt -weights models/resnet_scale/resnet18_nobat_scale.caffemodel -iterations 100
