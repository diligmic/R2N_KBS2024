####file to convert KBs from a R b to R(a,b)

import os

starting_path = "family_RLogic"

files = ["train","valid","test"]
for file in files:
    with open(os.path.join(starting_path,file+".txt")) as f:
        lines = f.readlines()
        lines_functional = [line.split()[1]+"("+line.split()[0]+","+line.split()[2]+")" for line in lines]
        new_file = open(os.path.join(starting_path,file+'functional.txt'), 'w+')
        for line in lines_functional:
            new_line_list = line+".\n"
            new_file.write(new_line_list)
        f.close()
