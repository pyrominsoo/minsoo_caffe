mkdir Bit_log
mkdir Bit_log/intbit
mkdir Bit_log/fracbit
cp ./include/minsoo/fracbit/fixed49_15.hpp ./include/minsoo/fixed.hpp;
make;
./examples/cifar10/test_full.sh 2> Bit_log/fracbit/fixed49_15.log
cp ./include/minsoo/intbit/fixed15_49.hpp ./include/minsoo/fixed.hpp;
make;
./examples/cifar10/test_full.sh 2> Bit_log/intbit/fixed15_49.log
cp ./include/minsoo/fracbit/fixed48_16.hpp ./include/minsoo/fixed.hpp;
make;
./examples/cifar10/test_full.sh 2> Bit_log/fracbit/fixed48_16.log
cp ./include/minsoo/intbit/fixed16_48.hpp ./include/minsoo/fixed.hpp;
make;
./examples/cifar10/test_full.sh 2> Bit_log/intbit/fixed16_48.log
