#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe test -model models/bvlc_reference_caffenet/minsoo_test.prototxt -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
