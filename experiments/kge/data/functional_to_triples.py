####file to convert KBs from R(a,b) to a R b

import os

starting_path = "data/countries"

files = ["countries_S1","countries_S2","countries_S3"]
for file in files:
    with open(os.path.join(starting_path,file+".txt")) as f:
        lines = f.readlines()
        lines = [line.replace("(","\t").replace(",","\t").replace(").","") for line in lines]
        new_file = open(os.path.join(starting_path,file+'_amieKB_to_process.txt'), 'w+')
        for line in lines:
            line_list = line.split()
            new_line_list = line_list[1] + " " + line_list[0] + " " + line_list[2] +"\n"
            new_file.write(new_line_list)
        f.close()


