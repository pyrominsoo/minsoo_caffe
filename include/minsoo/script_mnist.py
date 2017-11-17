
def file_gen():
    filename = "run_all.sh"
    f = open(filename,'w')
    f.write("mkdir Bit_log\n")
    f.write("mkdir Bit_log/intbit\n")
    f.write("mkdir Bit_log/fracbit\n")
    for i in range(2,17):
        fracbits = i;
        intbits = 64 - i;
        f.write("cp ./include/minsoo/fracbit/fixed" + str(intbits) + "_" + str(fracbits) + ".hpp ./include/minsoo/fixed.hpp;\n")
        f.write("make all;\n")
        f.write("./examples/mnist/test_lenet.sh 2> Bit_log/fracbit/fixed" + str(intbits) + "_" + str(fracbits) + ".log\n")
        
        intbits = i;
        fracbits = 64 - i;
        f.write("cp ./include/minsoo/intbit/fixed" + str(intbits) + "_" + str(fracbits) + ".hpp ./include/minsoo/fixed.hpp;\n")
        f.write("make all;\n")
        f.write("./examples/mnist/test_lenet.sh 2> Bit_log/intbit/fixed" + str(intbits) + "_" + str(fracbits) + ".log\n")

file_gen()
