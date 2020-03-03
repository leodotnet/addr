from collections import namedtuple
import pdb
import subprocess
from collections import namedtuple, defaultdict, OrderedDict
import os
import csv
import reader

chinese_digit = '一二三四五六七八九十百千万'

def seq2chunk(seq):
    chunks = []
    label = None
    last_label = None
    start_idx = 0
    for i in range(len(seq)):
        tok = seq[i]
        label = tok[2:] if tok.startswith('B') or tok.startswith('I') else tok
        if tok.startswith('B') or last_label != label:
            if last_label == None:
                start_idx = i
            else:
                chunks.append((last_label, start_idx, i))
                start_idx = i

        last_label = label

    chunks.append((label, start_idx, len(seq)))

    return chunks

# mylist = ['B-prov', 'I-prov', 'B-city', 'B-city', 'I-city', 'I-city', 'I-district', 'I-district', 'I-district', 'I-district', 'I-road', 'I-road', 'B-roadno', 'I-roadno']
# print(mylist)
# print(seq2chunk(mylist))

def count_common_chunks(chunk1, chunk2):
    common = 0
    for c1 in chunk1:
        for c2 in chunk2:
            if c1 == c2:
                common += 1

    return common


def get_performance(match_num, gold_num, pred_num):
    p = (match_num + 0.0) / pred_num
    r = (match_num + 0.0) / gold_num

    try:
        f1 = 2 * p * r / (p + r)
    except ZeroDivisionError:
        f1 = 0.0

    return p, r, f1

def action2chunk(actions:[]):
    chunks = []
    chunk = list()
    start_pos = 0
    pos = 0
    for action_tok in actions:

        if action_tok.startswith('APPEND'):
            pos += 1
        else: #LABEL(..)
            label_str = action_tok[6:-1]
            chunk = (label_str, start_pos, pos)
            chunks.append(chunk)
            start_pos = pos
    return chunks


def action2chunk_rbt(actions:[]):
    chunks = []
    chunk = list()
    start_pos = 0
    pos = 0

    label_str = None

    for action_tok in actions:
        if action_tok.startswith('APPEND'):
            pos += 1
        elif action_tok.startswith('SEPARATE'):
            if label_str != None:
                chunk = (label_str, start_pos, pos)
                chunks.append(chunk)

            label_str = None
        else: #LABEL(..)
            label_str = action_tok[6:-1]
            start_pos = pos

    return chunks



def chunk2seq(chunks:[]):
    seq = []
    for label, start_pos, end_pos in chunks:
        seq.append('B-' + label)
        for i in range(start_pos, end_pos - 1):
            seq.append('I-' + label)

    return seq



def get_action_seq2(raw_labels, vocab_action):
    action_seq = []
    label = None
    for i in range(len(raw_labels)):
        tok = raw_labels[i]
        if tok.startswith('B'):

            if label != None:
                #action_seq.append('SEPARATE')
                action_seq.append('LABEL(' + label + ')')

            label = tok[2:].lower()
            #action_seq.append('LABEL(' + label + ')')

        action_seq.append('APPEND')

    #action_seq.append('SEPARATE')
    action_seq.append('LABEL(' + label + ')')

    action_seq = [vocab_action.w2i[x] for x in action_seq]

    return action_seq


def get_action_seq(raw_labels, vocab_action):
    action_seq = []
    label = None
    for i in range(len(raw_labels)):
        tok = raw_labels[i]
        if tok.startswith('B'):

            if label != None:
                action_seq.append('SEPARATE')

            label = tok[2:].lower()
            action_seq.append('LABEL(' + label + ')')

        action_seq.append('APPEND')

    action_seq.append('SEPARATE')

    action_seq = [vocab_action.w2i[x] for x in action_seq]

    return action_seq


def get_action_seq_rbt(raw_labels, vocab_action):
    action_seq = []
    label = None
    n = 0

    chunks = seq2chunk(raw_labels)
    #print('chunks:',chunks)

    for i in range(len(chunks)):

        label, start_idx, end_idx = chunks[i]

        action_seq.append('LABEL(' + label + "'" + ')')
        action_seq.append('LABEL(' + label + ')')

        for j in range(start_idx, end_idx):
            action_seq.append('APPEND')

        action_seq.append('SEPARATE')

        if i < len(chunks) - 2:
            pass
        elif i == len(chunks) - 2:
            next_label, next_start_idx, next_end_idx = chunks[i + 1]

            action_seq.append('LABEL(' + next_label + ')')
            for j in range(next_start_idx, next_end_idx):
                action_seq.append('APPEND')

            action_seq.append('SEPARATE')
            break

    for i in range(len(chunks) - 1):
        action_seq.append('SEPARATE')

    action_seq = [vocab_action.w2i[x] for x in action_seq]

    return action_seq


def action2treestr(actions, raw_char):
    ret = ''
    pos = 0
    for action_tok in actions:
        if action_tok.startswith('LABEL('):
            ret += '(' + action_tok[6:-1] + ' '
        elif action_tok.startswith('SEP'):
            ret += ') '
        else:
            ret += raw_char[pos] + ' '
            pos += 1
    return ret

def is_digit(tok:str):

    tok = tok.strip()

    if tok.isdigit():
        return True


    if tok in chinese_digit:
        return True



    return False


def zero_normalize(tok:str):
    counter = 0
    new_tok = ''
    for i in range(len(tok)):
        if is_digit(tok[i]):
            new_tok += '0'
            counter += 1
        else:
            new_tok += tok[i]
    return new_tok, counter



def normalize(word):
    newword = ''
    for c in word:
        if is_digit(c):
            newword += '0'
        else:
            newword += c
    return newword


def eval_by_script(output_filename:str):
    cmdline = ["perl", "conlleval.pl"]
    cmd = subprocess.Popen(cmdline, stdin=open(output_filename, 'r', encoding='utf-8'),  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = cmd.communicate()
    print(stdout.decode("utf-8") )








