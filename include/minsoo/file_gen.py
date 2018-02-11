
def file_gen():
    for i in range(1,17):
        fracbits = i
        intbits = 32 - i
        filename = "fracbit/fixed" + str(intbits) + "_" + str(fracbits) + ".hpp"
        f = open(filename,'w')
        f.write("#define INTBITS " + str(intbits) + "\n")
        f.write("#define FRACBITS " + str(fracbits) + "\n")
        
        intbits = i
        fracbits = 32 - i
        filename = "intbit/fixed" + str(intbits) + "_" + str(fracbits) + ".hpp"
        f = open(filename,'w')
        f.write("#define INTBITS " + str(intbits) + "\n")
        f.write("#define FRACBITS " + str(fracbits) + "\n")

file_gen()
