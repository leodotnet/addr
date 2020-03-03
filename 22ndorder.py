def loadfile(filename):
    f = open(filename, 'r', encoding='utf-8')
    insts = []
    inst = list()

    for line in f:
        line = line.strip()
        if line == "":
            if inst != None:
                insts.append(inst)
            inst = list()
        else:
            inst.append(line.split())
    f.close()
    return insts

def writefile(filename, insts):
    f = open(filename, 'w', encoding='utf-8')

    for inst in insts:
        inst_str = '\n'.join([' '.join(item) for item in inst])
        f.write(inst_str + '\n\n')
    f.close()
    return insts


for filename in ['trial.txt', 'train.txt', 'dev.txt', 'test.txt']:
    insts = loadfile('data/' + filename)

    insts_2ndorder = []
    for inst in insts:
        inst_2ndorder = []
        size = len(inst)
        for pos in range(size):
            char, tag = inst[pos]
            _, next_tag = inst[pos + 1] if pos < size - 1 else (None, "<END>")
            new_tag = tag + ':' +next_tag
            inst_2ndorder.append((char, new_tag))
        insts_2ndorder.append(inst_2ndorder)

    outputfilename = filename.replace('.txt', '.order2.txt')
    writefile('data/' +outputfilename, insts_2ndorder)

