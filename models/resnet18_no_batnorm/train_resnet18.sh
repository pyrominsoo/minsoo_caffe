#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=models/resnet18_no_batnorm/solver.prototxt --weights=models/resnet18_no_batnorm/20190703/no_batnorm_iter_300000.caffemodel $@
