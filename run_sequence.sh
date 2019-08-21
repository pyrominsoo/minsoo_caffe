echo "1" > current_mult;
./test_resnet50.sh 2> float_20190820.acc;
mv winner.log float.log
echo "5" > current_mult;
echo "6" > DRUM_K;
./test_resnet50.sh 2> drum6_20190820.acc;
mv winner.log drum6.log
echo "6" > current_mult;
echo "7" > DRUM_K;
./test_resnet50.sh 2> mitchk7_20190820.acc;
mv winner.log mitchk7.log
echo "7" > current_mult;
echo "7" > DRUM_K;
./test_resnet50.sh 2> mitchk7_bias_20190820.acc;
mv winner.log mitchk7_bias.log
echo "8" > current_mult;
echo "7" > DRUM_K;
./test_resnet50.sh 2> mitchk7_bias_c1_20190820.acc;
mv winner.log mitchk7_bias_c1.log
echo "10" > current_mult;
echo "7" > DRUM_K;
./test_resnet50.sh 2> mitchk7_c1_20190820.acc;
mv winner.log mitchk7_c1.log

