
def file_gen():
    filename = "run_all_alex.sh"
    f = open(filename,'w')
    f.write("#!/bin/bash\n\n")
    f.write("export CAFFE_ROOT=$(pwd)\n")
    f.write("rm -rf Alex_log\n")
    f.write("mkdir Alex_log\n")
    f.write("mkdir Alex_log/intbit\n")
    f.write("mkdir Alex_log/fracbit\n")
    for i in range(6,11):
        fracbits = i;
        intbits = 32 - i;
        f.write("cp ./include/minsoo/fracbit/fixed" + str(intbits) + "_" + str(fracbits) + ".hpp ./include/minsoo/fixed.hpp;\n")
        f.write("make;\n")
        f.write("./test_alexnet.sh 2> Alex_log/fracbit/fixed" + str(intbits) + "_" + str(fracbits) + ".log\n")
        
        intbits = i;
        fracbits = 32 - i;
        f.write("cp ./include/minsoo/intbit/fixed" + str(intbits) + "_" + str(fracbits) + ".hpp ./include/minsoo/fixed.hpp;\n")
        f.write("make;\n")
        f.write("./test_alexnet.sh 2> Alex_log/intbit/fixed" + str(intbits) + "_" + str(fracbits) + ".log\n")
    for i in range(2,6):
        fracbits = i;
        intbits = 32 - i;
        f.write("cp ./include/minsoo/fracbit/fixed" + str(intbits) + "_" + str(fracbits) + ".hpp ./include/minsoo/fixed.hpp;\n")
        f.write("make;\n")
        f.write("./test_alexnet.sh 2> Alex_log/fracbit/fixed" + str(intbits) + "_" + str(fracbits) + ".log\n")
        
        intbits = i;
        fracbits = 32 - i;
        f.write("cp ./include/minsoo/intbit/fixed" + str(intbits) + "_" + str(fracbits) + ".hpp ./include/minsoo/fixed.hpp;\n")
        f.write("make;\n")
        f.write("./test_alexnet.sh 2> Alex_log/intbit/fixed" + str(intbits) + "_" + str(fracbits) + ".log\n")
    for i in range(11,17):
        fracbits = i;
        intbits = 32 - i;
        f.write("cp ./include/minsoo/fracbit/fixed" + str(intbits) + "_" + str(fracbits) + ".hpp ./include/minsoo/fixed.hpp;\n")
        f.write("make;\n")
        f.write("./test_alexnet.sh 2> Alex_log/fracbit/fixed" + str(intbits) + "_" + str(fracbits) + ".log\n")
        
        intbits = i;
        fracbits = 32 - i;
        f.write("cp ./include/minsoo/intbit/fixed" + str(intbits) + "_" + str(fracbits) + ".hpp ./include/minsoo/fixed.hpp;\n")
        f.write("make;\n")
        f.write("./test_alexnet.sh 2> Alex_log/intbit/fixed" + str(intbits) + "_" + str(fracbits) + ".log\n")

file_gen()
