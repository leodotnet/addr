import os
import numpy as np
import pickle
import pdb
import torch
import string
from tqdm import tqdm
from gensim.models import KeyedVectors
from collections import namedtuple, defaultdict, OrderedDict
from termcolor import colored
import re
import jieba

from optparse import OptionParser
#from preprocesstwitter import tokenize
import random

usage = "dataanlysis.py --input [inputfile]"

parser = OptionParser(usage=usage)
parser.add_option("--input", type="string", help="inputfile", default="train.txt", dest="inputfile")
parser.add_option("--type", type="string", help="type", default="poi", dest="type")
parser.add_option("--outputpath", type="string", help="output", default="output", dest="outputpath")

(options, args) = parser.parse_args()



def get_spans(output : list) -> list:

    spans = []
    size = len(output)

    for i in range(size):
        tag, tag_type = output[i][0], output[i][2:]
        if tag.startswith('B'):
            spans.append((i, -1, tag_type))

    for i in range(len(spans)):
        start, _, tag_type = spans[i]

        pos = start
        while(pos + 1 < size):
            if not output[pos + 1].startswith('I'):
                break
            else:
                pos += 1

        spans[i] = (start, pos, tag_type.lower())

    return spans


if __name__ == "__main__":
    f = open(options.inputfile, 'r', encoding='utf-8')

    print('Start Data analysis')
    print(options)

    insts = []
    inst = list()

    for line in f:
        line = line.strip()
        if line == "":
            if inst != None:
                insts.append(inst)
            inst = list()
        else:
            inst.append(line.split(' '))
    f.close()

    print('insts:', len(insts), 'insts[0]:', insts[0])

    total = 0
    stats = defaultdict(int)
    span_length = defaultdict(list)
    label = defaultdict(int)

    distinct_label = defaultdict(set)

    for inst in insts:

        tuple_size = len(inst[0])

        if tuple_size == 2:
            tokens, golds = list(zip(*inst))
        else:
            tokens, postags, golds = list(zip(*inst))

        gold_spans = get_spans(golds)



        for gold_span in gold_spans:

            spantext = ''.join(tokens[gold_span[0]: gold_span[1] + 1])
            distinct_label[gold_span[2]].add(spantext)

            if gold_span[2] == options.type or options.type == "none":
                spantext = ''.join(tokens[gold_span[0] : gold_span[1] + 1])
                length = len(spantext)

                # if length >= 9:
                #     length = 900

                stats[length] += 1
                total += 1

                span_length[length].append(''.join(spantext))
                label[gold_span[2]] += 1



    print('Data Stats:')


    #print(stats)
    span_list = []
    for k, v in stats.items():
        span_list.append((k,v))

    span_list = sorted(span_list, key=lambda x:x[0], reverse=False)

    print(colored('total:\t' + str(total), 'yellow'))
    for k, v in span_list:
        #print(str(k) +':\t' + str(v * 100.0 / total) + '\t' + str(', '.join(span_length[k][:5])))
        print("{0}:\t{1:.2f}%\t{2}, ...".format(k, v * 100 / total, ', '.join(span_length[k][:10])))

    print()
    print()

    label_list = []
    for k, v in label.items():
        label_list.append((k, v))

    label_list = sorted(label_list, key=lambda x: x[1], reverse=True)

    print(colored('total:\t' + str(total), 'yellow'))
    for k, v in label_list:
        # print(str(k) +':\t' + str(v * 100.0 / total) + '\t' + str(', '.join(span_length[k][:5])))
        print("{0}:\t{1:.2f}%, \t{2} ...".format(k, v * 100 / total, v))

    # f = open(options.type + ".txt", "w", encoding='utf-8')
    # for k, v in span_list:
    #     f.write(','.join(span_length[k]))
    #     f.write('\n')
    # f.close()

    print(colored('Distinct:', 'yellow'))
    distinct_total = 0
    for k, v in distinct_label.items():
        print(k,':\t', len(v), (str(v) if len(v) < 10 else ''))
        distinct_total += len(v)
    print('Distinct Chunk:', distinct_total)
    exit()



    f = open(options.outputpath + '/' + options.type + ".txt", "w", encoding='utf-8')
    for k, v in span_list:
        output_list = []
        for item in span_length[k]:

            segs = list(jieba.cut(item, cut_all=False))
            suffix =  segs[-1]
            output_list.append(suffix)

            # if k == 4:
            #     suffix = item[-2:]
            #     output_list.append(suffix)
            # elif k == 3:
            #     suffix = item[-1:]
            #     output_list.append(suffix)
            #
            #     suffix = item
            #     output_list.append(suffix)
            #
            # elif k in [5,6,7]:
            #     suffix = item[-1:]
            #     output_list.append(suffix)
            #
            #     suffix = item[-2:]
            #     output_list.append(suffix)
            #
            #     suffix = item[-3:]
            #     output_list.append(suffix)
            #
            # elif k == 2:
            #     suffix = item
            #     output_list.append(suffix)
            # elif k in [8, 9, 10, 11, 12, 13, 14]:
            #     suffix = item[-2:]
            #     output_list.append(suffix)
            #
            #     suffix = item[-3:]
            #     output_list.append(suffix)
            # elif k in [16, 17, 18]:
            #     suffix = item[-2:]
            #     output_list.append(suffix)


        output_set = set()
        for item in output_list:
            ret = re.search(r'\d+', item)
            if ret == None:
                ret = re.search(r'[A-Za-z0-9_]', item)
                if ret == None:
                    output_set.add(item)
        f.write(','.join(output_set) + '\n')
    f.close()

