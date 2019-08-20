echo "10" > current_mult;
echo "7" > DRUM_K;
./test_resnet50.sh 2> mitchk7_c1_20190727.acc;
echo "8" > current_mult;
echo "7" > DRUM_K;
./test_resnet50.sh 2> mitchk7_bias_c1_20190727.acc;
echo "7" > current_mult;
echo "7" > DRUM_K;
./test_resnet50.sh 2> mitchk7_bias_20190727.acc;
