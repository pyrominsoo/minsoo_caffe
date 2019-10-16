#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe test -gpu 0 -model models/bvlc_alexnet/train_val.prototxt -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel -iterations 100
