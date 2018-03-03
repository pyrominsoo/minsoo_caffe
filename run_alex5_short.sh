#!/bin/bash

export CAFFE_ROOT=$(pwd)
echo 6 > DRUM_K;
./test_alexnet.sh 2> alex5_short_mitchk6_1616_20180302.log;
echo 5 > DRUM_K;
./test_alexnet.sh 2> alex5_short_mitchk5_1616_20180302.log;
echo 4 > DRUM_K;
./test_alexnet.sh 2> alex5_short_mitchk4_1616_20180302.log;
echo 1 > DRUM_K;
./test_alexnet.sh 2> alex5_short_mitchk1_1616_20180302.log;
echo 2 > DRUM_K;
./test_alexnet.sh 2> alex5_short_mitchk2_1616_20180302.log;
echo 3 > DRUM_K;
./test_alexnet.sh 2> alex5_short_mitchk3_1616_20180302.log;
echo 7 > DRUM_K;
./test_alexnet.sh 2> alex5_short_mitchk7_1616_20180302.log;
echo 8 > DRUM_K;
./test_alexnet.sh 2> alex5_short_mitchk8_1616_20180302.log;
