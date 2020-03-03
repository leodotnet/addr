import util
import os
from optparse import OptionParser
from termcolor import colored
import random

usage = 'chunk2tree.py --inputdir [] --outputdir []'
parser = OptionParser(usage=usage)
parser.add_option("--inputdir", type="string", help="inputdir", default="data", dest="inputdir")
parser.add_option("--outputdir", type="string", help="outputdir", default="data", dest="outputdir")
parser.add_option("--type", type="string", help="type", default="NRBT", dest="type")
parser.add_option("--ordertype", type=int, help="ordertype", default=0, dest="ordertype")


(options, args) = parser.parse_args()



label2id = {}
id2label = {}


ASSIST = 'assist'
REDUNDANT = 'redundant'
NO = 'no'
ROOT = 'S'

def non_terminal_label(label):
    if label[-1] == "'":
        return label
    else:
        return label + "'"

def terminal_label(label):
    if label[-1] == "'":
        return label[:-1]
    else:
        return label

def get_parent_label(child1 : str, child2 : str, order_type = 0):
    parent = None

    if order_type == 1:
        return get_parent_label_reverse(child1, child2)

    if order_type == 2:
        return get_parent_label_order(child1, child2)

    if order_type == 3:
        return get_parent_label_empty(child1, child2)

    if order_type == 4:
        return get_parent_label_semi(child1, child2)

    if child1.startswith(REDUNDANT) or child2.startswith(REDUNDANT):
       if child1.startswith(REDUNDANT) and child2.startswith(REDUNDANT):
           parent = non_terminal_label(REDUNDANT)
       elif child1.startswith(REDUNDANT):
           parent = non_terminal_label(child2)
       else:
           parent = non_terminal_label(child1)

    elif child1.startswith(ASSIST) or child2.startswith(ASSIST):
       if child1.startswith(ASSIST) and child2.startswith(ASSIST):
           parent = non_terminal_label(ASSIST)
       elif child1.startswith(ASSIST):
           parent = non_terminal_label(child2)
       else:
           parent = non_terminal_label(child1)


    else:

        last1_priority_id = label2id[terminal_label(child1)]
        last2_priority_id = label2id[terminal_label(child2)]


        if last1_priority_id >= last2_priority_id:
            parent = non_terminal_label(child2)
        else:
            parent = non_terminal_label(child1)



    return parent


def get_parent_label_reverse(child1 : str, child2 : str):
    last1_priority_id = label2id[terminal_label(child1)]
    last2_priority_id = label2id[terminal_label(child2)]

    if last1_priority_id <= last2_priority_id:
        parent = non_terminal_label(child2)
    else:
        parent = non_terminal_label(child1)

    return parent



def get_parent_label_order(child1 : str, child2 : str):
    last1_priority_id = label2id[terminal_label(child1)]
    last2_priority_id = label2id[terminal_label(child2)]

    if last1_priority_id >= last2_priority_id:
        parent = non_terminal_label(child2)
    else:
        parent = non_terminal_label(child1)

    return parent


def get_parent_label_empty(child1 : str, child2 : str):
    return non_terminal_label("Y")

def get_parent_label_semi(child1 : str, child2 : str):
    return non_terminal_label(child1)


def read_file(filename):
    f = open(filename, 'r', encoding='utf-8')
    insts = []
    inst = list()
    num_inst = 0
    for line in f:
        line = line.strip()
        if line == "":
            if inst != None:

                inst = [tuple(x) for x in inst]

                tmp = list(zip(*inst))

                x = tmp[0]
                y = [x[:2] + x[2:].lower() for x in tmp[1]]

                chunks = util.seq2chunk(y)

                insts.append((x, chunks))

            inst = list()
        else:
            inst.append(line.split())
    f.close()

    return insts


def chunk2str(chunk, text):
    label_str, start_idx, end_idx = chunk
    ret = '(' + label_str + ' ' + ' '.join(['(XX ' + x + ')' for x in text[start_idx:end_idx]]) + ' )'
    return ret


def chunks2RBTree(insts, rule, rule_root, is_train = True):

    trees = []

    if is_train:
        for text, chunks in insts:

            last_chunk = chunks[-1]

            tree_str = chunk2str(last_chunk, text)


            if len(chunks) == 1:

                tree_str = '(S ' + tree_str + ' )'


            else:

                for i in reversed(range(len(chunks) - 1)):

                    second_last_chunk = chunks[i]

                    tmp_tree_str = chunk2str(second_last_chunk, text)

                    children = second_last_chunk[0] + ',' + last_chunk[0]
                    parent_label_str = rule[children] if i > 0 else rule_root[children]
                    #parent_label_str = get_parent_label(second_last_chunk[0], last_chunk[0]) if i > 0 else ROOT

                    tree_str = '(' + parent_label_str + ' ' + tmp_tree_str + ' ' + tree_str + ' )'

                    last_chunk = (parent_label_str, second_last_chunk[1], last_chunk[2])


            #print(tree_str)
            trees.append(tree_str)



    else:
        for text, chunks in insts:
            tree_str = '(S '
            for chunk in chunks:
                tree_str += chunk2str(chunk ,text) + ' '
            tree_str += ')'
            trees.append(tree_str)


    return trees



def chunks2LBTree(insts, rule, rule_root, is_train = True):

    trees = []

    if is_train:
        for text, chunks in insts:


            first_chunk = chunks[0]

            tree_str = chunk2str(first_chunk, text)

            if len(chunks) == 1:
                tree_str = '(S ' + tree_str + ' )'

                # print(colored(text, 'red'))
                # print(colored(str(chunks), 'red'))
                # print(colored(tree_str, 'red'))
                # print()

            else:

                for i in range(1, len(chunks)):

                    second_chunk = chunks[i]

                    tmp_tree_str = chunk2str(second_chunk, text)

                    children = first_chunk[0] + ',' + second_chunk[0]
                    #try:
                    parent_label_str = rule[children] if i < len(chunks) - 1 else rule_root[children]
                    #parent_label_str = get_parent_label(second_last_chunk[0], last_chunk[0]) if i > 0 else ROOT
                    # except:
                    #     print(text)
                    #     print('i:', i, ' children:', children)
                    #     print('chunks:\n', chunks)
                    #     exit()

                    tree_str = '(' + parent_label_str + ' ' + tmp_tree_str + ' ' + tree_str + ' )'

                    first_chunk = (parent_label_str, first_chunk[1], second_chunk[2])


            #print(tree_str)
            trees.append(tree_str)

    else:
        for text, chunks in insts:
            tree_str = '(S '
            for chunk in chunks:
                tree_str += chunk2str(chunk ,text) + ' '
            tree_str += ')'
            trees.append(tree_str)


    return trees



def chunks2NRBTree(insts, is_train = True, order_type = 0):

    trees = []

    if is_train:
        for text, chunks in insts:


            if len(chunks) == 1:

                last_chunk = chunks[-1]

                tree_str = chunk2str(last_chunk, text)
            else:

                last_chunk = chunks[-1]

                tree_str = chunk2str(last_chunk, text)


                for i in reversed(range(len(chunks) - 1)):

                    second_last_chunk = chunks[i]

                    tmp_tree_str = chunk2str(second_last_chunk, text)

                    children = second_last_chunk[0] + ',' + last_chunk[0]
                    #parent_label_str = rule[children] if i > 0 else rule_root[children]
                    parent_label_str = get_parent_label(second_last_chunk[0], last_chunk[0], order_type)

                    tree_str = '(' + parent_label_str + ' ' + tmp_tree_str + ' ' + tree_str + ' )'

                    last_chunk = (parent_label_str, second_last_chunk[1], last_chunk[2])


            #print(tree_str)
            trees.append(tree_str)

    else:
        for text, chunks in insts:
            tree_str = "(country' "
            for chunk in chunks:
                tree_str += chunk2str(chunk ,text) + ' '
            tree_str += ')'
            trees.append(tree_str)


    return trees



def chunks2NLBTree(insts, is_train = True, order_type = 0):

    trees = []

    if is_train:
        for text, chunks in insts:


            if len(chunks) == 1:

                first_chunk = chunks[-1]

                tree_str = chunk2str(first_chunk, text)
            else:

                first_chunk = chunks[0]

                tree_str = chunk2str(first_chunk, text)


                for i in range(1, len(chunks)):

                    second_chunk = chunks[i]

                    tmp_tree_str = chunk2str(second_chunk, text)


                    parent_label_str = get_parent_label(first_chunk[0], second_chunk[0], order_type)

                    tree_str = '(' + parent_label_str + ' ' + tree_str + ' ' + tmp_tree_str + ' )'

                    first_chunk = (parent_label_str, first_chunk[1], second_chunk[2])


            #print(tree_str)
            trees.append(tree_str)

    else:
        for text, chunks in insts:
            tree_str = "(country' "
            for chunk in chunks:
                tree_str += chunk2str(chunk ,text) + ' '
            tree_str += ')'
            trees.append(tree_str)


    return trees




def chunks2HTree(insts, is_train = True):

    trees = []

    if is_train:
        for text, chunks in insts:


            if len(chunks) == 1:

                last_chunk = chunks[-1]

                tree_str = chunk2str(last_chunk, text)
            else:

                last_chunk = chunks[-1]

                tree_str = chunk2str(last_chunk, text)


                for i in reversed(range(len(chunks) - 1)):

                    second_last_chunk = chunks[i]

                    if (second_last_chunk[0] == 'roadno' and chunks[i - 1][0] == 'road') or (second_last_chunk[0] == 'subroadno' and chunks[i - 1][0] == 'subroad'):
                        third_last_chunk = chunks[i - 1]
                        third_last_chunk_str = chunk2str(third_last_chunk, text)
                        second_last_chunk_str = chunk2str(second_last_chunk, text)
                        parent_label_str = get_parent_label(third_last_chunk[0], second_last_chunk[0])
                        tmp_tree_str = '(' + parent_label_str + ' ' +third_last_chunk_str + ' ' + second_last_chunk_str + ' )'
                        second_last_chunk = (parent_label_str, third_last_chunk[1], second_last_chunk[2])
                        i -= 1

                    tmp_tree_str = chunk2str(second_last_chunk, text)

                    children = second_last_chunk[0] + ',' + last_chunk[0]
                    #parent_label_str = rule[children] if i > 0 else rule_root[children]
                    parent_label_str = get_parent_label(second_last_chunk[0], last_chunk[0])

                    tree_str = '(' + parent_label_str + ' ' + tmp_tree_str + ' ' + tree_str + ' )'

                    last_chunk = (parent_label_str, second_last_chunk[1], last_chunk[2])


            #print(tree_str)
            trees.append(tree_str)



    else:
        for text, chunks in insts:
            tree_str = "(country' "
            for chunk in chunks:
                tree_str += chunk2str(chunk ,text) + ' '
            tree_str += ')'
            trees.append(tree_str)


    return trees


def read_rules(filename):
    rule = {}
    rule_root = {}

    print('Loading Rule from ', filename, ' ...')
    f = open(filename, 'r', encoding='utf-8')

    for line in f:
        lhs, rhs = line.strip().split('->')
        if lhs == 'S':
            rule_root[rhs] = lhs
        else:
            rule[rhs] = lhs

    f.close()

    return rule, rule_root

label_list = []
ordertype = options.ordertype

label_filename = 'data/labels.txt'

if ordertype == 2:
    label_filename = 'data/labels_random.txt'

f = open(label_filename, 'r', encoding='utf-8')
for line in f:
    line = line.strip()
    id = len(label2id)
    label2id[line] = id
    id2label[id] = line
    label_list.append(line)
f.close()

# random.shuffle(label_list)
# f = open('data/labels_random.txt', 'w', encoding='utf-8')
# for label in label_list:
#     f.write(label + '\n')
# f.close()
# exit()

line = 'no'
id = len(label2id)
label2id[line] = id
id2label[id] = line

if options.type in ['RBT', 'LBT']:
    rule, rule_root = read_rules(os.path.join(options.inputdir, options.type + '.rules.txt'))



for filename in ['train.txt', 'dev.txt', 'test.txt', 'trial.txt']:

    print('Read ' + filename + ' ...')

    insts = read_file(os.path.join(options.inputdir, filename))

    if options.type == 'RBT':
        insts = chunks2RBTree(insts, rule, rule_root, filename.startswith('train'))
    elif options.type == 'NRBT':
        insts = chunks2NRBTree(insts, filename.startswith('train'), ordertype)
    elif options.type == 'NLBT':
        insts = chunks2NLBTree(insts, filename.startswith('train'), ordertype)
    elif options.type == 'HT':
        insts = chunks2HTree(insts, filename.startswith('train'))
    elif options.type == 'LBT':
        insts = chunks2LBTree(insts, rule, rule_root, filename.startswith('train'))


    treetype = options.type

    if ordertype == 1:  #Reverse
        treetype = 'R' + treetype
    elif ordertype == 2:
        treetype = 'RND' + treetype  #RNDdomly Shuffled
    elif ordertype == 3:
        treetype = 'EPT' + treetype  #RNDdomly Shuffled  Empty
    elif ordertype == 4:
        treetype = 'SEMI' + treetype  # RNDdomly Shuffled

    output_filename = os.path.join(options.outputdir, filename.replace('.txt', '.' + treetype + '.txt'))

    print('exporting ' + output_filename + ' ...')

    fout = open(output_filename, 'w', encoding='utf-8')

    fout.write('\n'.join(insts))
    fout.write('\n')

    fout.close()
