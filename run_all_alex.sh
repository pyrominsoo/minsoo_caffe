#!/bin/bash

export CAFFE_ROOT=$(pwd)
rm -rf Alex_log
mkdir Alex_log
mkdir Alex_log/intbit
mkdir Alex_log/fracbit
cp ./include/minsoo/fracbit/fixed57_7.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/fracbit/fixed57_7.log
cp ./include/minsoo/intbit/fixed7_57.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/intbit/fixed7_57.log
cp ./include/minsoo/fracbit/fixed56_8.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/fracbit/fixed56_8.log
cp ./include/minsoo/intbit/fixed8_56.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/intbit/fixed8_56.log
cp ./include/minsoo/fracbit/fixed55_9.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/fracbit/fixed55_9.log
cp ./include/minsoo/intbit/fixed9_55.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/intbit/fixed9_55.log
cp ./include/minsoo/fracbit/fixed54_10.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/fracbit/fixed54_10.log
cp ./include/minsoo/intbit/fixed10_54.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/intbit/fixed10_54.log
cp ./include/minsoo/fracbit/fixed53_11.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/fracbit/fixed53_11.log
cp ./include/minsoo/intbit/fixed11_53.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/intbit/fixed11_53.log
cp ./include/minsoo/fracbit/fixed52_12.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/fracbit/fixed52_12.log
cp ./include/minsoo/intbit/fixed12_52.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/intbit/fixed12_52.log
