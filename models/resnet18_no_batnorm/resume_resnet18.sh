#!/usr/bin/env sh
set -e

./build/tools/caffe train \
    --solver=models/resnet18/solver.prototxt \
    --snapshot=models/resnet18/resnet18_iter_255000.solverstate \
    $@
