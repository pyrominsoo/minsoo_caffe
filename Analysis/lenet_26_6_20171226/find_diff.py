import string
import sys

def find_diff(file_name0, file_name1, file_name2):
    file0 = open(file_name0, "r")
    file1 = open(file_name1, "r")
    file2 = open(file_name2, "r")
    outfile0 = open("fixed_diff.log", "w")
    outfile1 = open("mitch_diff.log", "w")
    outfile2 = open("float_diff.log", "w")

    line_num = 0
    line0 = file0.readline()
    line1 = file1.readline()
    line2 = file2.readline()

    line0_split = string.split(line0)
    line1_split = string.split(line1)
    keyword0 = line0_split[0]
    keyword1 = line1_split[0]
    if (keyword0 != "stat"):
        print "no stat at the beginning(0)"
        sys.exit()
    elif (keyword1 != "stat"):
        print "no stat at the beginning(1)"
        sys.exit()
    
    line0 = file0.readline()
    line1 = file1.readline()
    line2 = file2.readline()
    line0_split = string.split(line0)
    line1_split = string.split(line1)
    keyword0 = line0_split[0]
    keyword1 = line1_split[0]
    if (keyword0 != "pos"):
        print "no pos at the statis(0)"
        sys.exit()
    elif (keyword1 != "pos"):
        print "no pos at the statis(1)"
        sys.exit()
    num_batch0 = int(line0_split[1])
    num_batch1 = int(line1_split[1])
    if (num_batch0 != num_batch1):
        print "batch size 0 != 1"
        sys.exit()

    line0 = file0.readline()
    line1 = file1.readline()
    line2 = file2.readline()
    line0_split = string.split(line0)
    line1_split = string.split(line1)
    numcor0 = int(line0_split[1])
    numcor1 = int(line1_split[1])


    line0 = file0.readline()
    line1 = file1.readline()
    line2 = file2.readline()
    line0_split = string.split(line0)
    line1_split = string.split(line1)
    total0 = int(line0_split[1])
    total1 = int(line1_split[1])

    line0 = file0.readline()
    line1 = file1.readline()
    line2 = file2.readline()
    line0_split = string.split(line0)
    line1_split = string.split(line1)
    acc0 = float(line0_split[1])
    acc1 = float(line1_split[1])

    for idx_bat in range(num_batch0):
        line0 = file0.readline()
        line1 = file1.readline()
        line2 = file2.readline()
        line0_split = string.split(line0)
        line1_split = string.split(line1)
        keyword0 = line0_split[0]
        keyword1 = line1_split[0]
        if (keyword0 != "bat"):
            print "no bat at the batch " + str(idx_bat)
            sys.exit()
        elif (keyword1 != "bat"):
            print "no bat at the batch " + str(idx_bat)
            sys.exit()
        batch_id0 = int(line0_split[1])
        batch_id1 = int(line1_split[1])

        line0 = file0.readline()
        line1 = file1.readline()
        line2 = file2.readline()
        line0_split = string.split(line0)
        line1_split = string.split(line1)
        keyword0 = line0_split[0]
        keyword1 = line1_split[0]
        if (keyword0 != "pos"):
            print "no pos at the batch " + str(idx_bat)
            sys.exit()
        elif (keyword1 != "pos"):
            print "no pos at the batch" + str(idx_bat)
            sys.exit()
        num_infer0 = int(line0_split[1])
        num_infer1 = int(line1_split[1])
        if (num_infer0 != num_infer1):
            print "num infer 0 != 1"
            sys.exit()
        
        for idx_infer in range(num_infer0):
            line0 = file0.readline()
            line1 = file1.readline()
            line2 = file2.readline()
            line0_split = string.split(line0)
            line1_split = string.split(line1)
            keyword0 = line0_split[0]
            keyword1 = line1_split[0]
            if (keyword0 != "inf"):
                print "no inf at the inf " + str(idx_infer)
                sys.exit()
            elif (keyword1 != "inf"):
                print "no bat at the inf " + str(idx_infer)
                sys.exit()
            infer_id0 = int(line0_split[1])
            infer_id1 = int(line1_split[1])
            
            line0 = file0.readline()
            line1 = file1.readline()
            line2 = file2.readline()
            line0_split = string.split(line0)
            line1_split = string.split(line1)
            keyword0 = line0_split[0]
            keyword1 = line1_split[0]
            if (keyword0 != "cor"):
                print "no cor at the inf " + str(idx_infer)
                sys.exit()
            elif (keyword1 != "cor"):
                print "no cor at the inf " + str(idx_infer)
                sys.exit()
            cor0 = int(line0_split[1])
            cor1 = int(line1_split[1])
            if (cor0 != cor1):
                write_on = 1
            else:
                write_on = 0
            
            if (write_on) :
                write_string = "Infer: " 
                infer_num = idx_bat*num_infer0+idx_infer
                write_string += str(infer_num)
                write_string += "\n"
                outfile0.write(write_string)
                outfile1.write(write_string)
                outfile2.write(write_string)
                write_string = "Cor0: " + str(cor0) + " | Cor1: " + str(cor1)
                write_string += "\n"
                outfile0.write(write_string)
                outfile1.write(write_string)
                outfile2.write(write_string)
                
            line0 = file0.readline()
            line1 = file1.readline()
            line2 = file2.readline()
            line0_split = string.split(line0)
            line1_split = string.split(line1)
            keyword0 = line0_split[0]
            keyword1 = line1_split[0]
            if (keyword0 != "cpos"):
                print "no cpos at the inf " + str(idx_infer)
                sys.exit()
            elif (keyword1 != "cpos"):
                print "no cpos at the inf " + str(idx_infer)
                sys.exit()
            num_c0 = int(line0_split[1])
            num_c1 = int(line1_split[1])
            

            line0 = file0.readline()
            line1 = file1.readline()
            line2 = file2.readline()
            line0_split = string.split(line0)
            line1_split = string.split(line1)
            keyword0 = line0_split[0]
            keyword1 = line1_split[0]
            if (keyword0 != "ipos"):
                print "no ipos at the inf " + str(idx_infer)
                sys.exit()
            elif (keyword1 != "ipos"):
                print "no ipos at the inf " + str(idx_infer)
                sys.exit()
            num_i0 = int(line0_split[1])
            num_i1 = int(line1_split[1])

            for idx_c in range(num_c0):
                line0 = file0.readline()
                line1 = file1.readline()
                line2 = file2.readline()
                line0_split = string.split(line0)
                line1_split = string.split(line1)
                keyword0 = line0_split[0]
                keyword1 = line1_split[0]
                if (keyword0 != "cl"):
                    print "no cl at the cl " + str(idx_c)
                    sys.exit()
                elif (keyword1 != "cl"):
                    print "no cl at the cl " + str(idx_c)
                    sys.exit()
                cl_id0 = int(line0_split[1])
                cl_id1 = int(line1_split[1])
                
                line0 = file0.readline()
                line1 = file1.readline()
                line2 = file2.readline()
                line0_split = string.split(line0)
                line1_split = string.split(line1)
                keyword0 = line0_split[0]
                keyword1 = line1_split[0]
                if (keyword0 != "pos"):
                    print "no pos at the cl " + str(idx_c)
                    sys.exit()
                elif (keyword1 != "pos"):
                    print "no pos at the cl " + str(idx_c)
                    sys.exit()
                num_chan0 = int(line0_split[1])
                num_chan1 = int(line1_split[1])
                    
                if (write_on) :
                    write_string = "ConvLayer: " 
                    write_string += str(idx_c)
                    write_string += "\n"
                    outfile0.write(write_string)
                    outfile1.write(write_string)
                    outfile2.write(write_string)

                for idx_chan in range(num_chan0):
                    line0 = file0.readline()
                    line1 = file1.readline()
                    line2 = file2.readline()
                    line0_split = string.split(line0)
                    line1_split = string.split(line1)
                    keyword0 = line0_split[0]
                    keyword1 = line1_split[0]
                    if (keyword0 != "cc"):
                        print "no cc at the cc " + str(idx_chan)
                        sys.exit()
                    elif (keyword1 != "cc"):
                        print "no cc at the cc " + str(idx_chan)
                        sys.exit()
                    cc_id0 = int(line0_split[1])
                    cc_id1 = int(line1_split[1])
                
                    line0 = file0.readline()
                    line1 = file1.readline()
                    line2 = file2.readline()
                    line0_split = string.split(line0)
                    line1_split = string.split(line1)
                    keyword0 = line0_split[0]
                    keyword1 = line1_split[0]
                    if (keyword0 != "pos"):
                        print "no pos at the cc " + str(idx_chan)
                        sys.exit()
                    elif (keyword1 != "pos"):
                        print "no pos at the cc " + str(idx_chan)
                        sys.exit()
                    num_val0 = int(line0_split[1])
                    num_val0 = int(line1_split[1])
                
                    if (write_on) :
                        write_string = "ConvChannel: " 
                        write_string += str(idx_chan)
                        write_string += "\n"
                        outfile0.write(write_string)
                        outfile1.write(write_string)
                        outfile2.write(write_string)
                    
                    for idx_val in range(num_val0):
                        line0 = file0.readline()
                        line1 = file1.readline()
                        line2 = file2.readline()
                        if (write_on) :
                            outfile0.write(line0)
                            outfile1.write(line1)
                            outfile2.write(line2)
                        
            for idx_i in range(num_i0):
                line0 = file0.readline()
                line1 = file1.readline()
                line2 = file2.readline()
                line0_split = string.split(line0)
                line1_split = string.split(line1)
                keyword0 = line0_split[0]
                keyword1 = line1_split[0]
                if (keyword0 != "il"):
                    print "no il at the il " + str(idx_i)
                    sys.exit()
                elif (keyword1 != "il"):
                    print "no il at the il " + str(idx_i)
                    sys.exit()
                il_id0 = int(line0_split[1])
                il_id1 = int(line1_split[1])
                
                line0 = file0.readline()
                line1 = file1.readline()
                line2 = file2.readline()
                line0_split = string.split(line0)
                line1_split = string.split(line1)
                keyword0 = line0_split[0]
                keyword1 = line1_split[0]
                if (keyword0 != "pos"):
                    print "no pos at the il " + str(idx_i)
                    sys.exit()
                elif (keyword1 != "pos"):
                    print "no pos at the il " + str(idx_i)
                    sys.exit()
                num_val0 = int(line0_split[1])
                num_val1 = int(line1_split[1])
                
                if (write_on) :
                    write_string = "IPLayer: " 
                    write_string += str(idx_i)
                    write_string += "\n"
                    outfile0.write(write_string)
                    outfile1.write(write_string)
                    outfile2.write(write_string)
                
                for idx_val in range(num_val0):
                    line0 = file0.readline()
                    line1 = file1.readline()
                    line2 = file2.readline()
                    val0 = float(line0)
                    val1 = float(line1)
                    if (write_on) :
                        outfile0.write(line0)
                        outfile1.write(line1)
                        outfile2.write(line2)

    
find_diff("fixed.log", "mitchell.log", "float.log")
