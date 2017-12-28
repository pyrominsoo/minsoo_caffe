import string
import sys

def remove_0(file_name0, file_name1):
    file0 = open(file_name0, "r")
    file1 = open(file_name1, "r")
    outfile0 = open("fixed_no0.log", "w")
    outfile1 = open("mitch_no0.log", "w")

    line_num = 0

    while True:
        aline0 = file0.readline()
        aline1 = file1.readline()

        if not aline0: break

        line0_split = string.split(aline0)
        line1_split = string.split(aline1)
        keyword0 = line0_split[0]
        keyword1 = line1_split[0]
        if (keyword0 != "A:"):
            print "no A: at the beginning(0)"
            sys.exit()
        elif (keyword1 != "A:"):
            print "no A: at the beginning(1)"
            sys.exit()
        
        a0 = float(line0_split[1])
        a1 = float(line1_split[1])

        bline0 = file0.readline()
        bline1 = file1.readline()

        line0_split = string.split(bline0)
        line1_split = string.split(bline1)
        keyword0 = line0_split[0]
        keyword1 = line1_split[0]
        if (keyword0 != "B:"):
            print "no B: at the beginning(0)"
            sys.exit()
        elif (keyword1 != "B:"):
            print "no B: at the beginning(1)"
            sys.exit()
        b0 = float(line0_split[1])
        b1 = float(line1_split[1])
        
        line0 = file0.readline()
        line1 = file1.readline()
        line0_split = string.split(line0)
        line1_split = string.split(line1)
        keyword0 = line0_split[0]
        keyword1 = line1_split[0]
        if (keyword0 != "Answer:"):
            print "no Answer: at the beginning(0)"
            sys.exit()
        elif (keyword1 != "Answer:"):
            print "no Answer: at the beginning(1)"
            sys.exit()
        answer0 = float(line0_split[1])
        answer1 = float(line1_split[1])

        if (answer0 == 0 and answer1 == 0):
            continue
        else:
            outfile0.write(aline0)
            outfile0.write(bline0)
            outfile0.write(line0)
            outfile1.write(aline1)
            outfile1.write(bline1)
            outfile1.write(line1)

    
remove_0("gemm_fixed.log", "gemm_mitch.log")
