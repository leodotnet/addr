from operator import itemgetter
from itertools import count
from collections import Counter, defaultdict
import random
import dynet as dy
import numpy as np
import re
import os
import pickle
import pdb
import copy
import time
from reader import Reader, Vocab
from tp import TransitionParser, LSTM, BiLSTM_CRF, TransitionParser2, TransitionParser_RBT, BiLSTM_SEMICRF

from optparse import OptionParser
import random
#from PYEVALB import scorer
#from PYEVALB import parser as evalbparser
#from termcolor import colored
import util


usage = "addr.py --train [inputfile] --test"

parser = OptionParser(usage=usage)
parser.add_option("--data", type="string", help="data", default="data", dest="data")
parser.add_option("--train", type="string", help="train", default="train.txt", dest="train")
parser.add_option("--dev", type="string", help="dev", default="dev.txt", dest="dev")
parser.add_option("--test", type="string", help="test", default="test.txt", dest="test")
parser.add_option("--outputdir", type="string", help="outputdir", default="output", dest="outputdir")
parser.add_option("--emb", type="string", help="emb", default="glove", dest="emb")
parser.add_option("--task", type="string", help="task", default="train", dest="task")
parser.add_option("--epoch", type="int", help="epoch", default=50, dest="epoch")
parser.add_option("--trial", type="int", help="trial", default=0, dest="trial")
parser.add_option("--numtrain", type="int", help="numtrain", default=-1, dest="numtrain")
parser.add_option("--numtest", type="int", help="numtest", default=-1, dest="numtest")
parser.add_option("--debug", type="int", help="debug", default=0, dest="debug")
parser.add_option("--os", type="string", help="os", default="osx", dest="os")
parser.add_option("--check_every", type="int", help="check_every", default=1000, dest="check_every")
parser.add_option("--decay_patience", type="int", help="decay_patience", default=3, dest="decay_patience")
parser.add_option("--lr_patience", type="int", help="lr_patience", default=2, dest="lr_patience")
parser.add_option("--lr", type="float", help="lr", default=0.1, dest="lr")
parser.add_option("--optimizer", type="string", help="optimizer", default="sgd", dest="optimizer")
parser.add_option("--dropout", type="float", help="dropout", default=0.2, dest="dropout")
parser.add_option("--pretrain", type="string", help="pretrain", default="none", dest="pretrain")

parser.add_option("--WORD_DIM", type="int", help="WORD_DIM", default=100, dest="WORD_DIM")
parser.add_option("--LSTM_DIM", type="int", help="LSTM_DIM", default=200, dest="LSTM_DIM")
parser.add_option("--ACTION_DIM", type="int", help="ACTION_DIM", default=200, dest="ACTION_DIM")
parser.add_option("--NUM_LAYER", type="int", help="NUM_LAYER", default=2, dest="NUM_LAYER")

parser.add_option("--expname", type="string", help="expname", default="default", dest="expname")

parser.add_option("--trainsample", type="int", help="trainsample", default=-1, dest="trainsample")
parser.add_option("--autolr", type="int", help="autolr", default=0, dest="autolr")
parser.add_option("--normalize", type="int", help="normalize", default=1, dest="normalize")
parser.add_option("--model", type="string", help="model", default="TP", dest="model")
parser.add_option("--lexicon", type="string", help="lexicon", default="lexicon.txt", dest="lexicon")
parser.add_option("--lexiconepoch", type="int", help="lexiconepoch", default=0, dest="lexiconepoch")
parser.add_option("--syntactic_composition", type="int", help="syntactic_composition", default=1, dest="syntactic_composition")
parser.add_option("--singleton", type="int", help="singleton", default=1, dest="singleton")
parser.add_option("--MAXL", type="int", help="MAXL", default=7, dest="MAXL")
parser.add_option("--MAXLLIMIT", type="int", help="MAXLLIMIT", default=12, dest="MAXLLIMIT")

(options, args) = parser.parse_args()

print('Args:', options)
emb_path = {'giga': 'giga.vec' + str(options.WORD_DIM)}

if options.trial == 1:
    options.train = "trial.txt"
    options.dev = "trial.txt"
    options.test = "trial.txt"


Reader.load_lexicon(options.data, options.lexicon)

label_list = Vocab.read_vocab_label()

if options.model == 'TP_RBT':
    label_list = label_list + [x + "'" for x in label_list]

label_opening_list = ['(' + x for x in label_list]
vocab_label = Vocab.from_list(label_list)
vocab_char, vocab_char_count = Vocab.read_vocab_char_from_list([os.path.join(options.data, options.train), os.path.join(options.data, options.dev), os.path.join(options.data, options.test), os.path.join(options.data, options.lexicon)])   #os.path.join(options.data, options.train)

vocab_label_opening =  Vocab.from_list(label_opening_list)
action_list = ['APPEND', 'SEPARATE'] + ['LABEL(' + x + ')' for x in label_list]
if options.model == 'TP2':
    action_list = ['APPEND'] + ['LABEL(' + x + ')' for x in label_list]
vocab_action = Vocab.from_list(action_list)
NUM_ACTIONS = vocab_action.size()


if options.model == 'LSTM':
    label_list = ['B-' + x.lower() for x in label_list] + ['I-' + x.lower() for x in label_list]
    vocab_label = Vocab.from_list(label_list)
elif options.model == 'CRF':
    label_list = ['B-' + x.lower() for x in label_list] + ['I-' + x.lower() for x in label_list] + ['<START>', '<STOP>']
    vocab_label = Vocab.from_list(label_list)
elif options.model == 'SEMICRF':
    #label_list += ['<START>', '<STOP>']
    #label_list = ['town', 'community']
    vocab_label = Vocab.from_list(label_list)

print('label_list:',label_list)
reader = Reader(vocab_char, vocab_label, vocab_action, options)



# mylist = ['B-prov', 'I-prov', 'B-city', 'I-city', 'B-district', 'B-road']
# print('mylist:',mylist)
# print(colored('vocab_action:', 'red'),vocab_action.w2i["LABEL(prov')"])
# action_seq = util.get_action_seq_rbt(mylist, vocab_action)
# action_seq = [vocab_action.i2w[x] for x in action_seq]
# print('action_seq:',action_seq)
#
# print(colored('action2chunk_rbt:', 'red'), util.action2chunk_rbt(action_seq))
# exit()

train = reader.read_file(os.path.join(options.data, options.train), 'train')
dev = reader.read_file(os.path.join(options.data, options.dev),'eval')
test = reader.read_file(os.path.join(options.data, options.test),'eval')

if options.lexiconepoch > 0:
    lexicon = reader.read_file(os.path.join(options.data, options.lexicon),'train')

train = train[:options.numtrain]
dev = dev[:options.numtest]
test = test[:options.numtest]

print('Data Reading completed.')
print('len(train):', len(train), ' MAXL:', options.MAXL, end='')
print(' len(dev):', len(dev),' len(test):', len(test), end='')
if options.lexiconepoch > 0:
    print(' len(lexicon):', len(lexicon), end='')
print()

pretrain_emb = None
if options.pretrain != 'none':
    pretrain_emb = reader.load_pretrain(os.path.join(options.data, emb_path[options.pretrain]), True)


model_filename = os.path.join(options.outputdir, 'addr_' + options.expname + '.model')
dev_output = model_filename.replace('.model', '.dev.out')
test_output = model_filename.replace('.model', '.test.out')

print('The model/output will be saved as ', model_filename, dev_output, test_output)

model = dy.ParameterCollection()  # dy.Model()

if options.optimizer == 'adam':
    trainer = dy.AdamTrainer(model)
else:
    trainer = dy.SimpleSGDTrainer(model) #dy.AdamTrainer(model)

trainer.set_learning_rate(options.lr)

if options.model == 'TP' :
    parser = TransitionParser(model, options, vocab_char, vocab_char_count, vocab_label, vocab_label_opening, vocab_action, pretrain_emb)
elif options.model == 'TP2':
    parser = TransitionParser2(model, options, vocab_char, vocab_char_count, vocab_label, vocab_label_opening, vocab_action, pretrain_emb)
elif options.model == 'TP_RBT':
    parser = TransitionParser_RBT(model, options, vocab_char, vocab_char_count, vocab_label, vocab_label_opening, vocab_action, pretrain_emb)

elif options.model == 'LSTM':
    parser = LSTM(model, options, vocab_char, vocab_char_count, vocab_label, pretrain_emb)
elif options.model == 'CRF':
    parser = BiLSTM_CRF(model, options, vocab_char, vocab_char_count, vocab_label, pretrain_emb)
elif options.model == 'SEMICRF':
    options.MAXL = reader.options.MAXL
    print('MAXL is set to ', options.MAXL)
    parser = BiLSTM_SEMICRF(model, options, vocab_char, vocab_char_count, vocab_label, pretrain_emb)
else:
    print('Model ' + options.model + ' is not specified.')
    exit()


def train_lexicon(lexicon, max_epoch, sample = 1000):
    i = 0
    for epoch in range(max_epoch):
        words = 0
        total_loss = 0.0

        random.shuffle(lexicon)

        for inst in lexicon[:sample]:
            i += 1

            loss = parser.parse(inst, inst["action"])
            words += len(inst["raw_char"])
            if loss is not None:
                total_loss += loss.scalar_value()
                loss.backward()
                trainer.update()
            e = float(i) / sample

            if i % sample == 0:
                print('lexicon epoch {}: per-inst loss: {}'.format(e, total_loss / words), flush=True)
                words = 0
                total_loss = 0.0




def sampletrain(train, dev, test, sample = 100, shuffle = True):
    early_counter = 0
    decay_counter = 0

    best_f1 = 0
    # best_model = copy.deepcopy(model)

    i = 0
    print('Training starts...', flush=True)

    for epoch in range(options.epoch):
        words = 0
        total_loss = 0.0
        if shuffle:
            random.shuffle(train)
        totrain = train[:sample]
        for inst in totrain:
            i += 1

            loss = parser.parse(inst, inst["action"])
            words += len(inst["raw_char"])
            if loss is not None:
                total_loss += loss.scalar_value()
                loss.backward()
                trainer.update()
            e = float(i) / len(totrain)

            if i % len(totrain) == 0:
                print('epoch {}: per-inst loss: {}'.format(e, total_loss / words), flush=True)
                words = 0
                total_loss = 0.0

            if i >= len(totrain) * 2 and i % options.check_every == 0:
                p, r, f1 = parser.predict(dev, dev_output)
                if best_f1 < f1:
                    best_f1 = f1
                    # best_model = model
                    print('Better F1:', best_f1)
                    # best_model = copy.deepcopy(model)
                    print('Saving model to ', model_filename, '...')
                    model.save(model_filename)
                    early_counter = 0

                    print('On Test:')
                    p, r, f1 = parser.predict(test, test_output)

                else:
                    early_counter += 1
                    if early_counter > options.lr_patience:
                        decay_counter += 1
                        early_counter = 0
                        if decay_counter > options.decay_patience:
                            break
                        else:
                            # adjust_learning_rate(trainer)
                            cur_lr = trainer.learning_rate
                            # adj_lr = cur_lr / 2
                            adj_lr = cur_lr * 0.7
                            if options.autolr == 1:
                                trainer.set_learning_rate(adj_lr)
                                print("Adjust lr to ", adj_lr)

if options.task == 'train':

    if options.lexiconepoch > 0:
        train_lexicon(lexicon, options.lexiconepoch)

    sample = len(train) if options.trainsample == -1 else options.trainsample
    sampletrain(train, dev, test, sample, True)

print('Loading model from ',model_filename, ' ...')
model.populate(model_filename)
print('The model is loaded.')

print('Dev ', end='')
p, r, f1 = parser.predict(dev, dev_output)
#print('F1:', f1)
util.eval_by_script(dev_output)

print('Test ', end='')
p, r, f1 = parser.predict(test, test_output)
#print('F1:', f1)
util.eval_by_script(test_output)