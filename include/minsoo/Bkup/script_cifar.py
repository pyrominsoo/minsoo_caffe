
def file_gen():
    filename = "run_all_cifar.sh"
    f = open(filename,'w')
    f.write("mkdir Cifar_log\n")
    f.write("mkdir Cifar_log/intbit\n")
    f.write("mkdir Cifar_log/fracbit\n")
    for i in range(2,17):
        fracbits = i;
        intbits = 64 - i;
        f.write("cp ./include/minsoo/fracbit/fixed" + str(intbits) + "_" + str(fracbits) + ".hpp ./include/minsoo/fixed.hpp;\n")
        f.write("make;\n")
        f.write("./examples/cifar10/test_full.sh 2> Cifar_log/fracbit/fixed" + str(intbits) + "_" + str(fracbits) + ".log\n")
        
        intbits = i;
        fracbits = 64 - i;
        f.write("cp ./include/minsoo/intbit/fixed" + str(intbits) + "_" + str(fracbits) + ".hpp ./include/minsoo/fixed.hpp;\n")
        f.write("make;\n")
        f.write("./examples/cifar10/test_full.sh 2> Cifar_log/intbit/fixed" + str(intbits) + "_" + str(fracbits) + ".log\n")

file_gen()
