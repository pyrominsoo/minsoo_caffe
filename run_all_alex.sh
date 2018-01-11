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
cp ./include/minsoo/fracbit/fixed59_5.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/fracbit/fixed59_5.log
cp ./include/minsoo/intbit/fixed5_59.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/intbit/fixed5_59.log
cp ./include/minsoo/fracbit/fixed58_6.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/fracbit/fixed58_6.log
cp ./include/minsoo/intbit/fixed6_58.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/intbit/fixed6_58.log
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
cp ./include/minsoo/fracbit/fixed51_13.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/fracbit/fixed51_13.log
cp ./include/minsoo/intbit/fixed13_51.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/intbit/fixed13_51.log
cp ./include/minsoo/fracbit/fixed50_14.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/fracbit/fixed50_14.log
cp ./include/minsoo/intbit/fixed14_50.hpp ./include/minsoo/fixed.hpp;
make;
./test_alexnet.sh 2> Alex_log/intbit/fixed14_50.log
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
