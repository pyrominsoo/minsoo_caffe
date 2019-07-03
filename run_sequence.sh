echo "6" > current_mult;
echo "7" > DRUM_K;
./test_resnet50.sh 2> mitchk7_acc
mv winner.log mitchk7.log
echo "7" > current_mult;
echo "7" > DRUM_K;
./test_resnet50.sh 2> mitchk7_bias_acc;
mv winner.log mitchk7_bias.log
