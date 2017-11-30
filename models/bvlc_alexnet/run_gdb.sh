#!/usr/bin/env sh
set -e

TOOLS=./build/tools

gdb --args $TOOLS/caffe test -model models/bvlc_alexnet/train_val.prototxt -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel -iterations 2
