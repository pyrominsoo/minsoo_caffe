#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=models/resnet50_no_batnorm/solver.prototxt --weights=models/resnet50_no_batnorm/20190712/no_batnorm_iter_500000.caffemodel $@
