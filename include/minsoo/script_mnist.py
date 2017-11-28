
def file_gen():
    filename = "run_all_mnist.sh"
    f = open(filename,'w')
    f.write("mkdir MNIST_log\n")
    f.write("mkdir MNIST_log/intbit\n")
    f.write("mkdir MNIST_log/fracbit\n")
    for i in range(2,17):
        fracbits = i;
        intbits = 64 - i;
        f.write("cp ./include/minsoo/fracbit/fixed" + str(intbits) + "_" + str(fracbits) + ".hpp ./include/minsoo/fixed.hpp;\n")
        f.write("make all;\n")
        f.write("./examples/mnist/test_lenet.sh 2> MNIST_log/fracbit/fixed" + str(intbits) + "_" + str(fracbits) + ".log\n")
        
        intbits = i;
        fracbits = 64 - i;
        f.write("cp ./include/minsoo/intbit/fixed" + str(intbits) + "_" + str(fracbits) + ".hpp ./include/minsoo/fixed.hpp;\n")
        f.write("make all;\n")
        f.write("./examples/mnist/test_lenet.sh 2> MNIST_log/intbit/fixed" + str(intbits) + "_" + str(fracbits) + ".log\n")

file_gen()
