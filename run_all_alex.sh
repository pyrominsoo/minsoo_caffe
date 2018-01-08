#!/bin/bash

export CAFFE_ROOT=$(pwd)
rm -rf Alex_log
mkdir Alex_log
mkdir Alex_log/intbit
mkdir Alex_log/fracbit
cp ./include/minsoo/fracbit/fixed62_2.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/fracbit/fixed62_2.log
cp ./include/minsoo/intbit/fixed2_62.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/intbit/fixed2_62.log
cp ./include/minsoo/fracbit/fixed61_3.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/fracbit/fixed61_3.log
cp ./include/minsoo/intbit/fixed3_61.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/intbit/fixed3_61.log
cp ./include/minsoo/fracbit/fixed60_4.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/fracbit/fixed60_4.log
cp ./include/minsoo/intbit/fixed4_60.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/intbit/fixed4_60.log
cp ./include/minsoo/fracbit/fixed49_15.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/fracbit/fixed49_15.log
cp ./include/minsoo/intbit/fixed15_49.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/intbit/fixed15_49.log
cp ./include/minsoo/fracbit/fixed48_16.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/fracbit/fixed48_16.log
cp ./include/minsoo/intbit/fixed16_48.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/intbit/fixed16_48.log
