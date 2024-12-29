### FOR UniKER RULES
start_filepath = "MLN_rule.txt"
stop_filepath = "R2N_rule.txt"


UniKER_file = open(start_filepath, "r")
UniKER_rules = UniKER_file.readlines()
with open(stop_filepath, "w") as f:
    for idx, rule in enumerate(UniKER_rules):
        R2N_rule = "r"+str(idx)+":0:"
        rule_list = rule.split()
        rule_lenght = len(rule_list)
        if rule_lenght == 2:
            if rule_list[1][-2:] == "_v":
                R2N_rule += rule_list[1][:-2] + "(y,x)"
            else:
                R2N_rule += rule_list[1] + "(x,y)"
            R2N_rule += " -> " + rule_list[0] + "(x,y)"
        if rule_lenght == 3:
            if rule_list[1][-2:] == "_v":
                R2N_rule += rule_list[1][:-2] + "(y,x), "
            else:
                R2N_rule += rule_list[1] + "(x,y), "
            if rule_list[2][-2:] == "_v":
                R2N_rule += rule_list[2][:-2] + "(z,y)"
            else:
                R2N_rule += rule_list[2] + "(y,z)"
            R2N_rule += " -> " + rule_list[0] + "(x,z)"
        f.write(R2N_rule+"\n")
    f.close()


