import string
import sys

def collect_log():
    list_file_h = open("nonvarying_list","r")
    list_file = list_file_h.read().splitlines()
    out_file = open("result_out.txt", "w")

    n = 0
    for logfile_name in list_file:
        logfile_h = open(logfile_name, "r")
        logfile = logfile_h.read().splitlines()
        top_1 = 0
        top_5 = 0
        # Loss
        line = logfile[-1]
        line_split = string.split(line)
        loss = line_split[-2]
        # Accuracy
        line = logfile[-2]
        line_split = string.split(line)
        if "accuracy_top_5" in line:
            top_5 = line_split[-1]
            line = logfile[-3]
            line_split = string.split(line)
            top_1 = line_split[-1]
        else:
            top_5 = ""
            top_1 = line_split[-1]
        write_string = logfile_name + "\t" + str(loss) + "\t" + str(top_1) + "\t" 
        write_string = write_string + str(top_5) + "\n"
        out_file.write(write_string)
        logfile_h.close()
        n = n + 1
        print n
    out_file.close()
    list_file_h.close()

collect_log()
