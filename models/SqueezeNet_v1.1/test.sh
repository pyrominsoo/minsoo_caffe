#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe test -model models/SqueezeNet_v1.1/minsoo_test.prototxt -weights models/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel
