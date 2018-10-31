
echo "10" > current_mult;
./test_resnet50.sh 2> mitchk_c1.log;
mv mitchk_c1.log Log_resnet50/

echo "9" > current_mult;
./test_resnet50.sh 2> asm.log;
mv asm.log Log_resnet50/

echo "8" > current_mult;
./test_resnet50.sh 2> mitchk_bias_c1.log;
mv mitchk_bias_c1.log Log_resnet50/

echo "7" > current_mult;
./test_resnet50.sh 2> mitchk_bias.log;
mv mitchk_bias.log Log_resnet50/

echo "6" > current_mult;
./test_resnet50.sh 2> mitchk.log;
mv mitchk.log Log_resnet50/

echo "5" > current_mult;
./test_resnet50.sh 2> drum.log;
mv drum.log Log_resnet50/

echo "4" > current_mult;
./test_resnet50.sh 2> iterlog.log;
mv iterlog.log Log_resnet50/

echo "3" > current_mult;
./test_resnet50.sh 2> mitch.log;
mv mitch.log Log_resnet50/

echo "2" > current_mult;
./test_resnet50.sh 2> fixed.log;
mv fixed.log Log_resnet50/

echo "1" > current_mult;
./test_resnet50.sh 2> float.log;
mv float.log Log_resnet50/
