import numpy as np
import pickle
import pdb
import torch
import string
#from tqdm import tqdm
#import gensim
#from gensim.models import KeyedVectors
from collections import namedtuple, defaultdict, OrderedDict
import  util
import os
import csv

UNK = '</s>'
NUM = '<NUM>'

class Vocab:
    def __init__(self, w2i):
        self.w2i = dict(w2i)
        self.i2w = {i: w for w, i in w2i.items()}

    @classmethod
    def from_list(cls, words):
        w2i = {}
        idx = 0
        for word in words:
            w2i[word] = idx
            idx += 1
        return Vocab(w2i)

    @classmethod
    def read_vocab_char(cls, vocab_fname):
        words = set()
        words.add(NUM)
        words_count = defaultdict(int)
        with open(vocab_fname, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line == "":
                    continue

                char, label = line.split()
                if util.is_digit(char):
                    words_count[NUM] += 1


                words.add(char)
                words_count[char] += 1
        return  Vocab.from_list([UNK] + list(words)), words_count

    @classmethod
    def read_vocab_char_from_list(cls, vocab_fname_list):
        words = set()
        words.add(NUM)
        words_count = defaultdict(int)
        for vocab_fname in vocab_fname_list:
            with open(vocab_fname, 'r', encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()
                    if line == "":
                        continue

                    char, label = line.split()
                    if util.is_digit(char):
                        words_count[NUM] += 1

                    normalize, count = util.zero_normalize(char)
                    if count > 0:
                        words.add(normalize)
                        words_count[normalize] += 1

                    words.add(char)
                    words_count[char] += 1
        return Vocab.from_list([UNK] + list(words)), words_count

    @classmethod
    def read_vocab_label(self, filename = './data/labels.txt'):
        f = open(filename, 'r', encoding='utf-8')
        labels = []
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            else:
                labels.append(line)
        f.close()

        return labels

    def size(self):
        return len(self.w2i.keys())







class Reader():

    def __init__(self, vocab_char : Vocab, vocab_label: Vocab, vocab_action: Vocab, options):
        self.vocab_char = vocab_char
        self.vocab_label = vocab_label
        self.vocab_action = vocab_action
        self.options = options


    def read_file(self, filename, mode="train", order = 1):
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
                    inst = {}
                    inst["raw_char"] = tmp[0]
                    if order == 1:
                        inst["raw_label"] = [x[:2] + x[2:].lower() for x in tmp[1]]
                    elif order == 2:
                        inst["raw_label"] = tmp[1]

                    #Normalize all the numbers
                    if self.options.normalize == 1:
                        #tmp[0] = [NUM if util.is_digit(x) else x for x in tmp[0]] #[NUM if util.is_digit(x) else x for x in tmp[0]]
                        tmp[0] = [util.normalize(x) for x in tmp[0]]
                    inst["raw_char_normalized"] = tmp[0]

                    inst["char"] = [self.vocab_char.w2i[x] if x in self.vocab_char.w2i else self.vocab_char.w2i[UNK] for x in tmp[0]]

                    if self.options.model == 'LSTM' or self.options.model == 'CRF':
                        inst["label"] = [self.vocab_label.w2i[x] for x in inst["raw_label"] ] #
                        inst["action"] = None

                    if self.options.model == 'SEMICRF':
                        inst["action"] = None
                        if order == 1:
                            inst["chunk"] = [(self.vocab_label.w2i[tag], s, e) for tag, s,e in util.seq2chunk(inst["raw_label"])]
                        elif order == 2:
                            chunks = util.seq2chunk(inst["raw_label"])
                            size = len(chunks)
                            for pos in range(size):
                                tag, s, e = chunks[pos]
                                next_tag, _, _ = chunks[pos + 1] if pos < size - 1 else ("<end>", None, None)
                                l_o2 = tag.lower() + ':' + next_tag.lower()
                                chunks[pos] = (self.vocab_label.w2i[l_o2], s, e)
                            inst["chunk"] = chunks

                        L = max([e - s for tag, s,e in util.seq2chunk(inst["raw_label"]) ])

                        if L > self.options.MAXLLIMIT and mode == "train":
                            print('Discard: ', ''.join(inst["raw_char"]), ' L=', L)
                            inst = list()
                            continue

                        if L > self.options.MAXL and mode == "train":
                            self.options.MAXL = L


                    if self.options.model == 'TP2':
                        inst["action"] = util.get_action_seq2(inst["raw_label"], self.vocab_action)
                    elif self.options.model == 'TP':
                        inst["action"] = util.get_action_seq(inst["raw_label"], self.vocab_action)
                    elif self.options.model == 'TP_RBT':
                        inst["action"] = util.get_action_seq_rbt(inst["raw_label"], self.vocab_action)
                    insts.append(inst)

                inst = list()
            else:
                inst.append(line.split())
        f.close()

        return insts

    @classmethod
    def load_lexicon(cls, path ='data', outputfile = 'lexicon.txt'):

        name_mapping = {'provinces': 'prov', 'cities': 'city', 'areas': 'district', 'streets': 'town'}

        category = "provinces,cities,areas,streets"
        inputpath = "data/Administrative-divisions-of-China/dist/"

        place = defaultdict(set)
        for place_type in category.split(','):

            filename = os.path.join(inputpath, place_type + '.csv')

            with open(filename, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    keyword = row['name']
                    place[name_mapping[place_type]].add(keyword)

                f.close()

        fout = open(os.path.join(path, outputfile), 'w', encoding='utf-8')
        for label in place:
            for item in place[label]:
                chars = tuple(item)

                prefix = 'B'
                for char in chars:
                    fout.write(char + ' ' + prefix + '-' + label + '\n')
                    prefix = 'I'

                fout.write('\n')

        fout.close()

        return place


    def get_vocab(self, insts):
        vocab = set()
        for inst in insts:
            for char in inst["raw"]:
                if char not in vocab:
                    vocab.add(char)

                    newchar = ''
                    for c in char:
                        if util.is_digit(c):
                            newchar += '0'
                        else:
                            newchar += c
                    vocab.add(newchar)


        return Vocab.from_list([UNK] + list(vocab))


    def load_pretrain(self, filename : str, saveemb=True):
        print('Loading Pretrained Embedding from ', filename,' ...')
        vocab_dic = {}
        with open(filename, encoding='utf-8') as f:
            for line in f:
                s_s = line.split()
                if s_s[0] in self.vocab_char.w2i:
                    vocab_dic[s_s[0]] = np.array([float(x) for x in s_s[1:]])
                    # vocab_dic[s_s[0]] = [float(x) for x in s_s[1:]]

        unknowns = vocab_dic[UNK] #np.random.uniform(-0.01, 0.01, self.options.WORD_DIM).astype("float32")
        numbers = np.random.uniform(-0.01, 0.01, self.options.WORD_DIM).astype("float32")
        # ret_mat = []
        ret_mat = np.zeros((len(self.vocab_char.i2w), self.options.WORD_DIM))
        unk_counter = 0
        for token_id in self.vocab_char.i2w:
            token = self.vocab_char.i2w[token_id]
            if token in vocab_dic:
                # ret_mat.append(vocab_dic[token])
                ret_mat[token_id] = vocab_dic[token]
            elif util.is_digit(token) or token == '<NUM>':
                ret_mat[token_id] = numbers
            else:
                # ret_mat.append(unknowns)
                ret_mat[token_id] = unknowns
                # print "Unknown token:", token
                unk_counter += 1
                #print('unk:', token)
        ret_mat = np.array(ret_mat)

        print('ret_mat shape:', ret_mat.shape)

        if saveemb:
            with open('glove.emb', "wb") as f:
                pickle.dump(ret_mat, f)

        print("{0} unk out of {1} vocab".format(unk_counter, len(self.vocab_char.i2w)))
        print('Glove Embedding is loaded.')
        return ret_mat



