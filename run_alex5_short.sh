#!/bin/bash

export CAFFE_ROOT=$(pwd)
echo 6 > DRUM_K;
./test_alexnet.sh 2> alex5_short_mitchk6_bias_c1_1616_20180314.log;
echo 5 > DRUM_K;
./test_alexnet.sh 2> alex5_short_mitchk5_bias_c1_1616_20180314.log;
echo 4 > DRUM_K;
./test_alexnet.sh 2> alex5_short_mitchk4_bias_c1_1616_20180314.log;
echo 3 > DRUM_K;
./test_alexnet.sh 2> alex5_short_mitchk3_bias_c1_1616_20180314.log;
echo 1 > DRUM_K;
./test_alexnet.sh 2> alex5_short_mitchk1_bias_c1_1616_20180314.log;
echo 2 > DRUM_K;
./test_alexnet.sh 2> alex5_short_mitchk2_bias_c1_1616_20180314.log;
echo 7 > DRUM_K;
./test_alexnet.sh 2> alex5_short_mitchk7_bias_c1_1616_20180314.log;
echo 8 > DRUM_K;
./test_alexnet.sh 2> alex5_short_mitchk8_bias_c1_1616_20180314.log;
