#!/bin/bash

export CAFFE_ROOT=$(pwd)
rm -rf Resnet_log
mkdir Resnet_log
mkdir Resnet_log/intbit
mkdir Resnet_log/fracbit
cp ./include/minsoo/fracbit/fixed18_14.hpp ./include/minsoo/fixed.hpp;
make;
./test_resnet101.sh 2> Resnet_log/fracbit/fixed18_14.log
cp ./include/minsoo/intbit/fixed14_18.hpp ./include/minsoo/fixed.hpp;
make;
./test_resnet101.sh 2> Resnet_log/intbit/fixed14_18.log
cp ./include/minsoo/fracbit/fixed20_12.hpp ./include/minsoo/fixed.hpp;
make;
./test_resnet101.sh 2> Resnet_log/fracbit/fixed20_12.log
cp ./include/minsoo/intbit/fixed12_20.hpp ./include/minsoo/fixed.hpp;
make;
./test_resnet101.sh 2> Resnet_log/intbit/fixed12_20.log
cp ./include/minsoo/fracbit/fixed22_10.hpp ./include/minsoo/fixed.hpp;
make;
./test_resnet101.sh 2> Resnet_log/fracbit/fixed22_10.log
cp ./include/minsoo/intbit/fixed10_22.hpp ./include/minsoo/fixed.hpp;
make;
./test_resnet101.sh 2> Resnet_log/intbit/fixed10_22.log
cp ./include/minsoo/fracbit/fixed21_11.hpp ./include/minsoo/fixed.hpp;
make;
./test_resnet101.sh 2> Resnet_log/fracbit/fixed21_11.log
cp ./include/minsoo/intbit/fixed11_21.hpp ./include/minsoo/fixed.hpp;
make;
./test_resnet101.sh 2> Resnet_log/intbit/fixed11_21.log
cp ./include/minsoo/fracbit/fixed19_13.hpp ./include/minsoo/fixed.hpp;
make;
./test_resnet101.sh 2> Resnet_log/fracbit/fixed19_13.log
cp ./include/minsoo/intbit/fixed13_19.hpp ./include/minsoo/fixed.hpp;
make;
./test_resnet101.sh 2> Resnet_log/intbit/fixed13_19.log
cp ./include/minsoo/intbit/fixed15_17.hpp ./include/minsoo/fixed.hpp;
make;
./test_resnet101.sh 2> Resnet_log/intbit/fixed15_17.log
cp ./include/minsoo/fracbit/fixed17_15.hpp ./include/minsoo/fixed.hpp;
make;
./test_resnet101.sh 2> Resnet_log/fracbit/fixed17_15.log

cp ./include/minsoo/intbit/fixed16_16.hpp ./include/minsoo/fixed.hpp;
make;
