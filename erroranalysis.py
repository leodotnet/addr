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


from optparse import OptionParser
#from preprocesstwitter import tokenize
import random

usage = "erroranalysis.py --input [inputfile]"

parser = OptionParser(usage=usage)
parser.add_option("--input", type="string", help="inputfile", default="", dest="inputfile")
#parser.add_option("--output", type="string", help="outputfile", default="", dest="outputfile")
parser.add_option("--type", type="string", help="type", default="None", dest="type")

(options, args) = parser.parse_args()

def inst2coll(inst, sep = '\t'):
    l_str = list(map(lambda x: sep.join(x), inst))
    l_str = '\n'.join(l_str) + '\n'
    return l_str


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

        spans[i] = (start, pos, tag_type)

    return spans


if __name__ == "__main__":
    f = open(options.inputfile, 'r', encoding='utf-8')

    print('Start Error analysis')
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
            inst.append(line.split('\t'))
    f.close()


    error_insts = []
    stats = defaultdict(int)
    span_error = defaultdict(int)
    type_error = defaultdict(int)

    chunk_length_stats = defaultdict(int)
    chunk_length_set = set()
    total_gold_chunks = 0

    length_threshould = 12

    for inst in insts:
        error_instance = False

        for token, gold, pred in inst:
            tokens, golds, preds = list(zip(*inst))
            gold_spans = get_spans(golds)
            pred_spans = get_spans(preds)

            for gold_span in gold_spans:

                if options.type == gold_span[2] or options.type == "None" or (
                        options.type == "no" and gold_span[2].endswith("no")):
                    length = gold_span[1] - gold_span[0] + 1

                    if length >= length_threshould:
                        length = length_threshould * 100

                    chunk_length_stats[str(length) + '_gold'] += 1

                    chunk_length_set.add(length)

                    total_gold_chunks += 1

            for pred_span in pred_spans:
                if options.type == pred_span[2] or options.type == "None" or (
                        options.type == "no" and pred_span[2].endswith("no")):
                    length = pred_span[1] - pred_span[0] + 1

                    if length >= length_threshould:
                        length = length_threshould * 100

                    chunk_length_stats[str(length) + '_pred'] += 1

                    chunk_length_set.add(length)

            for gold_span in gold_spans:

                if options.type == gold_span[2] or options.type == "None" or (
                        options.type == "no" and gold_span[2].endswith("no")):

                    for pred_span in pred_spans:
                        if pred_span[0] == gold_span[0] and pred_span[1] == gold_span[1]:
                            if pred_span[2] == gold_span[2]:
                                length = gold_span[1] - gold_span[0] + 1

                                if length >= length_threshould:
                                    length = length_threshould * 100

                                chunk_length_stats[str(length) + '_match'] +=1





        for token, gold, pred in inst:

            if gold != pred:
                error_instance = True
                break

        if error_instance:
            tokens, golds, preds = list(zip(*inst))

            gold_spans = get_spans(golds)
            pred_spans = get_spans(preds)

            type_match = False

            for span in gold_spans:
                if options.type == span[2] or options.type == "None" or (options.type == "no" and span[2].endswith("no")):
                    type_match = True
                    break

            if not type_match:
                continue



            discard = True




            for gold_span in gold_spans:

                if options.type == gold_span[2] or options.type == "None" or (options.type == "no" and gold_span[2].endswith("no")):

                    # stats['#gold_span'] += 1
                    # stats['#pred_span'] += len(pred_spans)

                    span_match = False
                    span_and_type_match = False

                    pred_type = None

                    for pred_span in pred_spans:
                        if pred_span[0] == gold_span[0] and  pred_span[1] == gold_span[1]:

                            span_match = True

                            if  pred_span[2] == gold_span[2]:

                                span_and_type_match = True

                            else:

                                pred_type = pred_span[2]
                                pass
                                #print(inst2coll(inst), '\n')

                            break


                    if span_match:
                        if span_and_type_match:
                            pass

                        else:
                            discard = False
                            stats['span_match_but_type_notmatch'] += 1
                            #type_error[' < ' + gold_span[2] + '[span_match_but_type_notmatch]'] += 1
                            type_error[pred_type + ' < ' + gold_span[2]] += 1
                            #print(inst2coll(inst), '\n')

                    else:
                        discard = False
                        stats['even_span_notmatch'] += 1
                        #type_error[' < ' + gold_span[2] + '[even_span_notmatch]'] += 1
                        type_error[' < ' + gold_span[2]] += 1


                        #span_error[str(pred_span[0] - gold_span[0]) + ',' + str(pred_span[1] - gold_span[1])] += 1

            if not discard:
                error_insts.append(inst)





    type_error_list = []
    sum = stats['span_match_but_type_notmatch'] + stats['even_span_notmatch'] + 0.0

    print('Error Stats [#Erros {}]:'.format(int(sum)))
    for k, v in stats.items():
        print(k, ':{0:.2f}%'.format(v * 100 / sum))
    #print(stats)

    #print('Total #Error:', sum)
    for k,v in type_error.items():
        type_error_list.append((k, v, "{0:.2f}%".format((v * 100 / sum))))

    type_error_list = sorted(type_error_list, key= lambda x:x[1], reverse=True)
    print('Type Error List:')
    print(type_error_list)


    print('Length Analysis:')
    print(chunk_length_stats)


    for length in sorted(chunk_length_set):

        length = str(length)

        g = chunk_length_stats[length + '_gold']
        p = chunk_length_stats[length + '_pred']
        m = chunk_length_stats[length + '_match'] + 0.0

        if p != 0:
            prec = m / p
        else:
            prec = 0

        if g != 0:
            rec = m / g
        else:
            rec = 0

        if prec + rec != 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0

        print(length ,"&", g, "{0:.2f}%".format(g*100/total_gold_chunks) ," & {0:.2f} & {1:.2f} & {2:.2f}".format(prec * 100, rec * 100, f1 * 100), ' \\\\') #, "   {0} {1} {2}".format(p, g, int(m)))

    print()


    if options.inputfile.endswith('.txt'):
        outputfile = options.inputfile.replace('.txt', '.err.' + options.type)
    elif options.inputfile.endswith('.out'):
        outputfile = options.inputfile.replace('.out', '.err.' + options.type)
    else:
        print('Unknown extension')
        exit()
    f = open(outputfile, 'w', encoding='utf-8')
    print('#Error Insts:', len(error_insts))
    for inst in error_insts:
        inst_str = inst2coll(inst, '\t') + '\n'
        f.write(inst_str)
    f.close()

