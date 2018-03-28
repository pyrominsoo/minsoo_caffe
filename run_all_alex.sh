#!/bin/bash

export CAFFE_ROOT=$(pwd)
rm -rf Alex_log
mkdir Alex_log
mkdir Alex_log/intbit
mkdir Alex_log/fracbit
cp ./include/minsoo/fracbit/fixed26_6.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/fracbit/fixed26_6.log
cp ./include/minsoo/intbit/fixed6_26.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/intbit/fixed6_26.log
cp ./include/minsoo/fracbit/fixed25_7.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/fracbit/fixed25_7.log
cp ./include/minsoo/intbit/fixed7_25.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/intbit/fixed7_25.log
cp ./include/minsoo/fracbit/fixed24_8.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/fracbit/fixed24_8.log
cp ./include/minsoo/intbit/fixed8_24.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/intbit/fixed8_24.log
cp ./include/minsoo/fracbit/fixed23_9.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/fracbit/fixed23_9.log
cp ./include/minsoo/intbit/fixed9_23.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/intbit/fixed9_23.log
cp ./include/minsoo/fracbit/fixed22_10.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/fracbit/fixed22_10.log
cp ./include/minsoo/intbit/fixed10_22.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/intbit/fixed10_22.log
cp ./include/minsoo/fracbit/fixed27_5.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/fracbit/fixed27_5.log
cp ./include/minsoo/intbit/fixed5_27.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/intbit/fixed5_27.log
cp ./include/minsoo/fracbit/fixed21_11.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/fracbit/fixed21_11.log
cp ./include/minsoo/intbit/fixed11_21.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/intbit/fixed11_21.log
cp ./include/minsoo/fracbit/fixed20_12.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/fracbit/fixed20_12.log
cp ./include/minsoo/intbit/fixed12_20.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/intbit/fixed12_20.log
cp ./include/minsoo/fracbit/fixed19_13.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/fracbit/fixed19_13.log
cp ./include/minsoo/intbit/fixed13_19.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/intbit/fixed13_19.log
cp ./include/minsoo/fracbit/fixed18_14.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/fracbit/fixed18_14.log
cp ./include/minsoo/intbit/fixed14_18.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/intbit/fixed14_18.log
cp ./include/minsoo/fracbit/fixed17_15.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/fracbit/fixed17_15.log
cp ./include/minsoo/intbit/fixed15_17.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/intbit/fixed15_17.log
cp ./include/minsoo/fracbit/fixed16_16.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/fracbit/fixed16_16.log
