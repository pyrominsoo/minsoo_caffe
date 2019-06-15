#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=models/resnet18/solver.prototxt $@
