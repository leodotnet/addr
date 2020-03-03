from operator import itemgetter
from itertools import count
from collections import Counter, defaultdict
import random
import dynet as dy
import numpy as np
import math
import re
from reader import Vocab
import util
from termcolor import colored

IMPLICIT_REDUCE_AFTER_SHIFT = 0

APPEND = 0  #APP
SEPARATE = 1 #SEP
LABEL_BEGIN = 2  #LBL

LABEL_LIMIT = 100


class BiLSTM_SEMICRF:

    def __init__(self,model : dy.ParameterCollection(), options, vocab_char : Vocab, vocab_char_count : {}, vocab_label: Vocab, pretrain_emb : np.ndarray):
        self.vocab_char = vocab_char
        self.vocab_char_count = vocab_char_count
        self.vocab_label = vocab_label  # prov
        self.vocab_char_count = vocab_char_count
        self.options = options
        self.model = model
        self.MAXL = options.MAXL

        self.options = options
        LSTM_DIM = options.LSTM_DIM
        self.LSTM_DIM = LSTM_DIM
        WORD_DIM = options.WORD_DIM
        NUM_LAYER = options.NUM_LAYER
        NUM_WORDS = self.vocab_char.size()
        NUM_LABELS = self.vocab_label.size()
        self.tag_size = NUM_LABELS

        H2DIM = 64
        BIN_DIM = 8

        # LSTM parameters
        self.bi_lstm = dy.BiRNNBuilder(NUM_LAYER, WORD_DIM, LSTM_DIM, self.model, dy.LSTMBuilder)
        self.bi_lstm.set_dropout(self.options.dropout)

        # SEGLSTM parameters
        self.seg_fwd_lstm = dy.LSTMBuilder(NUM_LAYER, LSTM_DIM, LSTM_DIM, self.model)
        self.seg_fwd_lstm.set_dropout(self.options.dropout)
        self.seg_bwd_lstm = dy.LSTMBuilder(NUM_LAYER, LSTM_DIM, LSTM_DIM, self.model)
        self.seg_bwd_lstm.set_dropout(self.options.dropout)

        if self.options.syntactic_composition != 2:
            self.W_sm = model.add_parameters((LSTM_DIM, LSTM_DIM * 2))
        else:
            self.W_sm = model.add_parameters((LSTM_DIM, LSTM_DIM))




        self.b_sm = model.add_parameters(LSTM_DIM)

        self.W_cf = model.add_parameters((LSTM_DIM, LSTM_DIM))
        self.W_ce = model.add_parameters((LSTM_DIM, LSTM_DIM))





        self.W_h12h2 = model.add_parameters((H2DIM, LSTM_DIM))
        self.b_h2 = model.add_parameters((H2DIM))

        self.W_h22o = model.add_parameters((1, H2DIM))
        self.b_o = model.add_parameters(1)

        self.c_start = model.add_parameters(LSTM_DIM)
        self.c_end = model.add_parameters(LSTM_DIM)


        self.y_tag_lookup = model.add_lookup_parameters((LSTM_DIM, LSTM_DIM))
        self.W_y = model.add_parameters((LSTM_DIM, LSTM_DIM))

        self.duration_tag_lookup = model.add_lookup_parameters((options.MAXL + 1, BIN_DIM))
        self.W_du = model.add_parameters((LSTM_DIM, BIN_DIM))


        # Transition matrix for tagging layer, [i,j] is score of transitioning to i from j
        self.transitions = self.model.add_lookup_parameters((NUM_LABELS, NUM_LABELS))

        self.words_lookup = model.add_lookup_parameters((NUM_WORDS, WORD_DIM))
        if self.options.pretrain != 'none':
            print('Init Lookup table with pretrain emb')
            self.words_lookup.init_from_array(pretrain_emb)

    def set_dropout(self, p):
        self.bi_lstm.set_dropout(p)

    def disable_dropout(self):
        self.bi_lstm.disable_dropout()


    def build_tagging_graph(self, sentence):
        dy.renew_cg()

        # embeddings = [self.word_rep(w) for w in sentence]
        embeddings = [self.words_lookup[w] for w in sentence]

        lstm_out = self.bi_lstm.transduce(embeddings)

        W = dy.parameter(self.W_sm)
        b = dy.parameter(self.b_sm)

        W_cf = dy.parameter(self.W_cf)
        W_ce = dy.parameter(self.W_ce)
        W_y = dy.parameter(self.W_y)
        W_du = dy.parameter(self.W_du)

        W_h12h2 = dy.parameter(self.W_h12h2)
        b_h2 = dy.parameter(self.b_h2)

        W_h22o = dy.parameter(self.W_h22o)
        b_o = dy.parameter(self.b_o)

        c_start = dy.parameter(self.c_start)
        c_end = dy.parameter(self.c_end)



        spans = {}


        def duration_emb(dur, num_bins = self.options.MAXL + 1):
            max_bin = num_bins - 1
            # if dur:
            #     dur = int((math.log(dur) / math.log(1.6))) + 1;
            if dur > max_bin:
                dur = max_bin

            return self.duration_tag_lookup[dur]

        LSTM_DIM_HALF = self.LSTM_DIM / 2
        n = len(lstm_out)
        for i in range(n):
            for j in range(i, n):

                L = j - i + 1

                if L > self.options.MAXL:
                    continue

                du_emb = duration_emb(L)

                for tag in range(self.tag_size):

                    y = self.y_tag_lookup[tag]

                    span_embeddings = lstm_out[i:j+1]

                    if self.options.syntactic_composition == 1:
                        fw_init = self.seg_fwd_lstm.initial_state()
                        bw_init = self.seg_bwd_lstm.initial_state()


                        fwd = fw_init.transduce(span_embeddings)
                        #bwd = bw_init.transduce(span_embeddings[::-1])
                        bwd = bw_init.transduce(reversed(span_embeddings))

                        bi = dy.concatenate([fwd[-1], bwd[-1]])
                    elif self.options.syntactic_composition == 2:
                        if i == 0:
                            fwd = lstm_out[j][:LSTM_DIM_HALF]
                        else:
                            fwd = lstm_out[j][:LSTM_DIM_HALF] - lstm_out[i - 1][:LSTM_DIM_HALF]

                        if j == n - 1:
                            bwd = lstm_out[i][LSTM_DIM_HALF:]
                        else:

                            bwd = lstm_out[i][LSTM_DIM_HALF:] - lstm_out[j + 1][LSTM_DIM_HALF:]


                        bi = dy.concatenate([fwd, bwd])
                    else:
                        bi = dy.concatenate([span_embeddings[0], span_embeddings[-1]])


                    cf_embeding = c_start if (i == 0) else lstm_out[i - 1]

                    ce_embeding = c_end if (j == n - 1) else lstm_out[j + 1]

                    h1 = dy.rectify(dy.affine_transform([b, W, bi, W_y, y, W_du, du_emb, W_cf, cf_embeding, W_ce, ce_embeding]))
                    #h1 = dy.rectify(dy.affine_transform([b, W, bi, W_y, y, W_cf, cf_embeding, W_ce, ce_embeding]))

                    if self.options.dropout > 0:
                        h2 = dy.dropout(dy.rectify(dy.affine_transform([b_h2,W_h12h2, h1])), self.options.dropout)
                    else:
                        h2 = dy.rectify(dy.affine_transform([b_h2, W_h12h2, h1]))

                    o = dy.rectify(dy.affine_transform([b_o, W_h22o, h2]))

                    spans[(i, j, tag)] = o




        return spans

    def score_sentence(self, spans, chunks):  #chunks [(tag, s, e)]  [s,e)
        #assert len(observations) == len(tags)
        score_seq = [0]
        score = dy.scalarInput(0)
        tags = [tag for tag, _, _, in chunks]

        for i, chunk in enumerate(chunks):
            tag, s, e = chunk
            #score = score + dy.pick(self.transitions[tags[i + 1]], tags[i]) + dy.pick(obs, tags[i + 1])
            score = score + spans[(s, e - 1, tag)]
            # if i == 0:
            #     score = score + spans[(s, e - 1, tag)]
            # else:
            #     score = score + spans[(s, e - 1, tag)] #+ dy.pick(self.transitions[tag], tags[i - 1])

            #score_seq.append(score.value())
        #score = score #+ dy.pick(self.transitions[self.vocab_label.w2i["<STOP>"]], tags[-1])
        return score

    def viterbi_loss(self, sentence, chunks):
        spans = self.build_tagging_graph(sentence)
        viterbi_chunks, viterbi_score = self.viterbi_decoding(spans, sentence)
        if viterbi_chunks != chunks:
            gold_score = self.score_sentence(spans, chunks)
            return (viterbi_score - gold_score), viterbi_chunks
        else:
            return dy.scalarInput(0), viterbi_chunks

    def parse(self, inst, actions = None):  #neg_log_loss
        sentence =  inst["char"]
        chunks = inst["chunk"]
        spans = self.build_tagging_graph(sentence)
        gold_score = self.score_sentence(spans, chunks)
        forward_score = self.forward(spans, sentence)
        if self.options.trial == 1:
            print(colored('forward_score:', 'red'),forward_score.value(), colored('\tgold_score:', 'red'),gold_score.value())
        return forward_score - gold_score

    def forward(self, spans, sentence):
        '''
        :param spans:  span[(i,j)] = [i,j]
        :param sentence:
        :return:
        '''

        def log_sum_exp(scores):
            #return dy.logsumexp(scores.npvalue().tolist())
            npval = scores.npvalue()
            argmax_score = np.argmax(npval)
            max_score_expr = dy.pick(scores, argmax_score)
            max_score_expr_broadcast = dy.concatenate([max_score_expr] * npval.size)
            #return max_score_expr + dy.log(dy.sum_cols(dy.transpose(dy.exp(scores - max_score_expr_broadcast))))
            '''
            sum_cols(x) has been deprecated.
            Please use sum_dim(x, [1]) instead.
            '''
            return max_score_expr + dy.log(dy.sum_dim(dy.exp(scores - max_score_expr_broadcast), [0]))



        for_expr = []
        for pos in range(0, len(sentence)):

            alphas_t = []
            for next_tag in range(self.tag_size):

                next_tag_expr_L = []
                for L in range(1, self.MAXL + 1):

                    span_start_pos = pos - L
                    if span_start_pos < -1:
                        continue

                    #obs = spans[(span_start_pos + 1 - 1, pos - 1)]
                    #obs_broadcast = dy.concatenate([dy.pick(obs, next_tag)] * self.tag_size)
                    obs = dy.concatenate([spans[(span_start_pos + 1, pos, next_tag)]] * self.tag_size)

                    if span_start_pos + 1 == 0:
                        next_tag_expr_L.append(obs)
                    else:
                        # print('spans:', len(spans[(span_start_pos + 1 - 1, pos - 1, next_tag)].value()))
                        # print('obs_broadcast:',len(obs_broadcast.value()))
                        # print('for_expr:', len(for_expr[span_start_pos].value()))
                        next_tag_expr_L.append(for_expr[span_start_pos] + obs) #+ self.transitions[next_tag])
                    #next_tag_expr_L.append(for_expr[span_start_pos] + self.transitions[next_tag] + obs_broadcast)

                next_tag_expr = dy.concatenate(next_tag_expr_L)
                alphas_t.append(log_sum_exp(next_tag_expr))

            for_expr_new = dy.concatenate(alphas_t)
            for_expr.append(for_expr_new)

        terminal_expr = for_expr[-1] #+ self.transitions[self.vocab_label.w2i["<STOP>"]]
        alpha = log_sum_exp(terminal_expr)


        return alpha

    def viterbi_decoding(self, spans, sentence):
        backpointers = [(None,None)]
        for_expr = []

        for pos in range(0, len(sentence)):

            bptrs_t = []
            vvars_t = []


            for next_tag in range(self.tag_size):

                next_tag_expr = []
                next_tag_arr = []
                best_tag_id_arr = []
                best_tag_score_arr = []

                for L in range(1, self.MAXL + 1):
                    span_start_pos = pos - L
                    if span_start_pos < -1:
                        continue

                    #obs = dy.pick(spans[(span_start_pos + 1 - 1, pos - 1)], next_tag)
                    obs = spans[(span_start_pos + 1, pos, next_tag)]

                    #next_tag_expr.append(for_expr[span_start_pos] + self.transitions[next_tag] + obs)
                    if span_start_pos + 1 == 0:
                        next_tag_expr.append(obs)
                    else:
                        next_tag_expr.append(for_expr[span_start_pos] + obs) #+ self.transitions[next_tag])

                    next_tag_arr.append(next_tag_expr[-1].npvalue())
                    best_tag_id_arr.append(np.argmax(next_tag_arr[-1]))
                    best_tag_score_arr.append(np.max(next_tag_arr[-1]))


                best_tag_id_arr = np.asarray(best_tag_id_arr)
                best_L = np.argmax(np.asarray(best_tag_score_arr))

                best_tag_id = best_tag_id_arr[best_L]


                bptrs_t.append((best_tag_id, best_L + 1))
                vvars_t.append(dy.pick(next_tag_expr[best_L], best_tag_id))

            for_expr_new = dy.concatenate(vvars_t)
            for_expr.append(for_expr_new)
            backpointers.append(bptrs_t)

        # Perform final transition to terminal
        terminal_expr = for_expr[-1]
        terminal_arr = terminal_expr.npvalue()
        best_tag_id = np.argmax(terminal_arr)
        path_score = dy.pick(terminal_expr, best_tag_id)
        # Reverse over the backpointers to get the best path
        best_path = []  # Start with the tag that was best for terminal

        pos = len(sentence)

        bptrs_t = backpointers[pos]


        while pos >= 0:
            best_tag_id_current = best_tag_id
            best_tag_id, L = bptrs_t[best_tag_id]
            best_path.append((best_tag_id_current, pos - L, pos))
            pos = pos - L
            bptrs_t = backpointers[pos]

            if pos == 0:
                break

        #start = best_path.pop()  # Remove the start symbol
        best_path.reverse()


        assert pos == 0
        # Return best path and best path's score
        return best_path, path_score

    def decode(self, inst):
        sentence = inst["char"]
        spans = self.build_tagging_graph(sentence)
        best_chunk_path, path_score = self.viterbi_decoding(spans, sentence)
        if self.options.trial == 1:
            print('best_chunk_path:', best_chunk_path)
        best_chunk_path = [(self.vocab_label.i2w[tag], s, e ) for tag, s, e  in best_chunk_path]

        best_path = []

        for chunk in best_chunk_path:
            best_path.append('B-' + chunk[0])
            best_path += ['I-' +  chunk[0]] * (chunk[2] - chunk[1] - 1)

        return best_path

    def predict(self, insts, fname='test'):
        total = []

        fout = open(fname, 'w', encoding='utf-8')

        match_num = 0
        gold_num = 0
        pred_num = 0

        i = 0
        for inst in insts:

            if i % 10 == 0:
                print('.', end='', flush=True)


            input_seq = tuple(inst["raw_char"])
            gold_seq = tuple(inst["raw_label"])
            predict_seq = tuple(self.decode(inst))

            if self.options.trial == 1:
                print(inst["raw_char"])
                print(inst["raw_char_normalized"])
                print(gold_seq)
                print(predict_seq)
                print()

            gold_chunks = util.seq2chunk(gold_seq)
            predict_chunks = util.seq2chunk(predict_seq)

            if self.options.trial == 1:
                print(gold_chunks)
                print(predict_chunks)
                print()
                print()

            match_num += util.count_common_chunks(gold_chunks, predict_chunks)
            gold_num += len(gold_chunks)
            pred_num += len(predict_chunks)

            o_list = zip(input_seq, gold_seq, predict_seq)
            fout.write('\n'.join(['\t'.join(x) for x in o_list]))
            fout.write('\n')

            i += 1
        print()

        fout.close()
        if self.options.trial == 1:
            print((gold_num, pred_num, match_num))
        p, r, f1 = util.get_performance(match_num, gold_num, pred_num)
        print('P,R,F: [{0:.2f}, {1:.2f}, {2:.2f}]'.format(p * 100, r * 100, f1 * 100))
        print()

        return p, r, f1


class BiLSTM_CRF:

    def __init__(self,model : dy.ParameterCollection(), options, vocab_char : Vocab, vocab_char_count : {}, vocab_label: Vocab, pretrain_emb : np.ndarray):
        self.vocab_char = vocab_char
        self.vocab_char_count = vocab_char_count
        self.vocab_label = vocab_label  # prov
        self.vocab_char_count = vocab_char_count
        self.options = options
        self.model = model

        self.options = options
        LSTM_DIM = options.LSTM_DIM
        WORD_DIM = options.WORD_DIM
        NUM_LAYER = options.NUM_LAYER
        NUM_WORDS = self.vocab_char.size()
        NUM_LABELS = self.vocab_label.size()
        self.tag_size = NUM_LABELS

        # LSTM parameters
        self.bi_lstm = dy.BiRNNBuilder(NUM_LAYER, WORD_DIM, LSTM_DIM, self.model, dy.LSTMBuilder)
        self.bi_lstm.set_dropout(self.options.dropout)

        # Matrix that maps from Bi-LSTM output to num tags
        self.lstm_to_tags_params = self.model.add_parameters((NUM_LABELS, LSTM_DIM))
        self.lstm_to_tags_bias = self.model.add_parameters(NUM_LABELS)
        self.mlp_out = self.model.add_parameters((NUM_LABELS, NUM_LABELS))
        self.mlp_out_bias = self.model.add_parameters(NUM_LABELS)

        self.W_sm = model.add_parameters((NUM_LABELS, LSTM_DIM))
        self.b_sm = model.add_parameters(NUM_LABELS)

        # Transition matrix for tagging layer, [i,j] is score of transitioning to i from j
        self.transitions = self.model.add_lookup_parameters((NUM_LABELS, NUM_LABELS))

        self.words_lookup = model.add_lookup_parameters((NUM_WORDS, WORD_DIM))
        if self.options.pretrain != 'none':
            print('Init Lookup table with pretrain emb')
            self.words_lookup.init_from_array(pretrain_emb)

    def set_dropout(self, p):
        self.bi_lstm.set_dropout(p)

    def disable_dropout(self):
        self.bi_lstm.disable_dropout()

    def word_rep(self, word):
        """
        For rare words in the training data, we will use their morphemes
        to make their representation
        """
        if self.train_vocab_ctr[word] > 5:
            return self.words_lookup[word]
        else:
            # # Use morpheme embeddings
            # morpheme_decomp = self.morpheme_decomps[word]
            # rep = self.morpheme_lookup[morpheme_decomp[0]]
            # for m in morpheme_decomp[1:]:
            #     rep += self.morpheme_lookup[m]
            # if self.morpheme_projection is not None:
            #     rep = self.morpheme_projection * rep
            # if np.linalg.norm(rep.npvalue()) >= 50.0:
            #     # This is meant to handle things like URLs and weird tokens like !!!!!!!!!!!!!!!!!!!!!
            #     # that are splitting into a lot of morphemes, and their large representations are cause NaNs
            #     # TODO handle this in a better way.  Looks like all such inputs are either URLs, email addresses, or
            #     # long strings of a punctuation token when the decomposition is > 10
            #     return self.words_lookup[self.vocab_label.w2i["<UNK>"]]

            if random.random() > 0.5:
                rep = self.words_lookup[self.vocab_char.w2i["<UNK>"]]
            return rep

    def build_tagging_graph(self, sentence):
        dy.renew_cg()

        # embeddings = [self.word_rep(w) for w in sentence]
        embeddings = [self.words_lookup[w] for w in sentence]


        lstm_out = self.bi_lstm.transduce(embeddings)


        H = dy.parameter(self.lstm_to_tags_params)
        Hb = dy.parameter(self.lstm_to_tags_bias)
        O = dy.parameter(self.mlp_out)
        Ob = dy.parameter(self.mlp_out_bias)

        W = dy.parameter(self.W_sm)
        b = dy.parameter(self.b_sm)


        scores = [dy.affine_transform([b, W, x]) for x in lstm_out]

        #print([x.value() for x in scores])
        # scores = []
        # for rep in lstm_out:
        #     score_t = O * dy.tanh(H * rep + Hb) + Ob
        #     scores.append(score_t)

        return scores

    def score_sentence(self, observations, tags):
        assert len(observations) == len(tags)
        score_seq = [0]
        score = dy.scalarInput(0)
        tags = [self.vocab_label.w2i["<START>"]] + tags
        for i, obs in enumerate(observations):
            score = score + dy.pick(self.transitions[tags[i + 1]], tags[i]) + dy.pick(obs, tags[i + 1])
            score_seq.append(score.value())
        score = score + dy.pick(self.transitions[self.vocab_label.w2i["<STOP>"]], tags[-1])
        return score

    def viterbi_loss(self, sentence, tags):
        observations = self.build_tagging_graph(sentence)
        viterbi_tags, viterbi_score = self.viterbi_decoding(observations)
        if viterbi_tags != tags:
            gold_score = self.score_sentence(observations, tags)
            return (viterbi_score - gold_score), viterbi_tags
        else:
            return dy.scalarInput(0), viterbi_tags

    def parse(self, inst, actions = None):  #neg_log_loss
        sentence =  inst["char"]
        tags = inst["label"]
        observations = self.build_tagging_graph(sentence)
        gold_score = self.score_sentence(observations, tags)
        forward_score = self.forward(observations)
        return forward_score - gold_score

    def forward(self, observations):

        def log_sum_exp(scores):
            #return dy.logsumexp(scores.npvalue().tolist())
            npval = scores.npvalue()
            argmax_score = np.argmax(npval)
            max_score_expr = dy.pick(scores, argmax_score)
            max_score_expr_broadcast = dy.concatenate([max_score_expr] * self.tag_size)
            #return max_score_expr + dy.log(dy.sum_cols(dy.transpose(dy.exp(scores - max_score_expr_broadcast))))
            '''
            sum_cols(x) has been deprecated.
            Please use sum_dim(x, [1]) instead.
            '''
            return max_score_expr + dy.log(dy.sum_dim(dy.exp(scores - max_score_expr_broadcast), [0]))

        init_alphas = [-1e10] * self.tag_size
        init_alphas[self.vocab_label.w2i["<START>"]] = 0

        for_expr = dy.inputVector(init_alphas)
        for obs in observations:
            alphas_t = []
            for next_tag in range(self.tag_size):
                obs_broadcast = dy.concatenate([dy.pick(obs, next_tag)] * self.tag_size)
                next_tag_expr = for_expr + self.transitions[next_tag] + obs_broadcast
                alphas_t.append(log_sum_exp(next_tag_expr))
            for_expr = dy.concatenate(alphas_t)
        terminal_expr = for_expr + self.transitions[self.vocab_label.w2i["<STOP>"]]
        alpha = log_sum_exp(terminal_expr)



        return alpha

    def viterbi_decoding(self, observations):
        backpointers = []
        init_vvars = [-1e10] * self.tag_size
        init_vvars[self.vocab_label.w2i["<START>"]] = 0  # <Start> has all the probability
        for_expr = dy.inputVector(init_vvars)
        trans_exprs = [self.transitions[idx] for idx in range(self.tag_size)]
        for obs in observations:
            bptrs_t = []
            vvars_t = []
            for next_tag in range(self.tag_size):
                next_tag_expr = for_expr + trans_exprs[next_tag]
                next_tag_arr = next_tag_expr.npvalue()
                best_tag_id = np.argmax(next_tag_arr)
                bptrs_t.append(best_tag_id)
                vvars_t.append(dy.pick(next_tag_expr, best_tag_id))
            for_expr = dy.concatenate(vvars_t) + obs

            backpointers.append(bptrs_t)
        # Perform final transition to terminal
        terminal_expr = for_expr + trans_exprs[self.vocab_label.w2i["<STOP>"]]
        terminal_arr = terminal_expr.npvalue()
        best_tag_id = np.argmax(terminal_arr)
        path_score = dy.pick(terminal_expr, best_tag_id)
        # Reverse over the backpointers to get the best path
        best_path = [best_tag_id]  # Start with the tag that was best for terminal
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()  # Remove the start symbol
        best_path.reverse()
        assert start == self.vocab_label.w2i["<START>"]
        # Return best path and best path's score
        return best_path, path_score

    def decode(self, inst):
        sentence = inst["char"]
        observations = self.build_tagging_graph(sentence)
        best_path, path_score = self.viterbi_decoding(observations)
        best_path = [self.vocab_label.i2w[x] for x in best_path]
        #print('path_score:', path_score.value())
        return best_path

    def predict(self, insts, fname='test'):
        total = []

        # print('transition:')
        # for i in range(self.tag_size):
        #     from_tag = self.vocab_label.i2w[i]
        #     print(from_tag)
        #     print([x.value() for x in self.transitions[i]])


        fout = open(fname, 'w', encoding='utf-8')

        match_num = 0
        gold_num = 0
        pred_num = 0

        i = 0
        for inst in insts:

            if i % 10 == 0:
                print('.', end='', flush=True)


            input_seq = tuple(inst["raw_char"])
            gold_seq = tuple(inst["raw_label"])
            predict_seq = tuple(self.decode(inst))

            if self.options.trial == 1:
                print(inst["raw_char"])
                print(inst["raw_char_normalized"])
                print(gold_seq)
                print(predict_seq)
                print()

            gold_chunks = util.seq2chunk(gold_seq)
            predict_chunks = util.seq2chunk(predict_seq)

            if self.options.trial == 1:
                print(gold_chunks)
                print(predict_chunks)
                print()
                print()

            match_num += util.count_common_chunks(gold_chunks, predict_chunks)
            gold_num += len(gold_chunks)
            pred_num += len(predict_chunks)

            o_list = zip(input_seq, gold_seq, predict_seq)
            fout.write('\n'.join(['\t'.join(x) for x in o_list]))
            fout.write('\n\n')

            i += 1
        print()

        fout.close()
        if self.options.trial == 1:
            print((gold_num, pred_num, match_num))
        p, r, f1 = util.get_performance(match_num, gold_num, pred_num)
        print('P,R,F: [{0:.2f}, {1:.2f}, {2:.2f}]'.format(p * 100, r * 100, f1 * 100))
        print()

        return p, r, f1


class LSTM:
    def __init__(self,model : dy.ParameterCollection(), options, vocab_char : Vocab, vocab_char_count : {}, vocab_label: Vocab, pretrain_emb : np.ndarray):
        self.vocab_char = vocab_char
        self.vocab_char_count = vocab_char_count
        self.vocab_label = vocab_label  # prov
        self.vocab_char_count = vocab_char_count


        self.options = options
        LSTM_DIM = options.LSTM_DIM
        WORD_DIM = options.WORD_DIM
        NUM_LAYER = options.NUM_LAYER
        NUM_WORDS = self.vocab_char.size()
        NUM_LABELS = self.vocab_label.size()

        # Lookup parameters for word embeddings
        self.LOOKUP = model.add_lookup_parameters((NUM_WORDS, WORD_DIM))
        if self.options.pretrain != 'none':
            print('Init Lookup table with pretrain emb')
            self.LOOKUP.init_from_array(pretrain_emb)

        self.LSTM = dy.BiRNNBuilder(NUM_LAYER, WORD_DIM, LSTM_DIM, model, dy.LSTMBuilder)
        self.LSTM.set_dropout(options.dropout)

        self.W_sm = model.add_parameters((NUM_LABELS, LSTM_DIM))
        self.b_sm = model.add_parameters(NUM_LABELS)

    def decode(self, inst):
        self.predict_seq = []
        self.parse(inst, False)
        return self.predict_seq

    def parse(self, inst, is_training = True):
        this_sents = this_words = this_loss = this_correct = 0
        dy.renew_cg()

        # Transduce all batch elements with an LSTM
        word_reps = self.LSTM.transduce([self.LOOKUP[x] for x in inst["char"]])

        # Softmax scores
        W = dy.parameter(self.W_sm)
        b = dy.parameter(self.b_sm)
        scores = [dy.affine_transform([b, W, x]) for x in word_reps]

        #print('inst["label"]:',inst["label"])
        losses = [dy.pickneglogsoftmax(score, tag) for score, tag in zip(scores, inst["label"])]
        loss_exp = dy.esum(losses)

        if not is_training:
            predict_seq_id = [np.argmax(dy.softmax(score).npvalue()) for score in scores]
            self.predict_seq = [self.vocab_label.i2w[id] for id in predict_seq_id]

        # correct = [np.argmax(score.npvalue()) == tag for score, tag in zip(scores, tags)]
        # this_correct += sum(correct)

        #this_loss += loss_exp.scalar_value()
        #this_words += len(inst["char"])



        return loss_exp

    def predict(self, insts, fname='test'):
        total = []

        # gold_path = fname + '.gold.txt'
        # pred_path = fname + '.pred.txt'
        # fgold = open(gold_path, 'w', encoding='utf-8')
        # fpred = open(pred_path, 'w', encoding='utf-8')

        fout = open(fname, 'w', encoding='utf-8')

        match_num = 0
        gold_num = 0
        pred_num = 0

        i = 0
        for inst in insts:

            if i % 10 == 0:
                print('.', end='', flush=True)


            input_seq = tuple(inst["raw_char"])
            gold_seq = tuple(inst["raw_label"])
            predict_seq = tuple(self.decode(inst))

            if self.options.trial == 1:
                print(inst["raw_char"])
                print(inst["raw_char_normalized"])
                print(gold_seq)
                print(predict_seq)
                print()

            gold_chunks = util.seq2chunk(gold_seq)
            predict_chunks = util.seq2chunk(predict_seq)

            if self.options.trial == 1:
                print(gold_chunks)
                print(predict_chunks)
                print()
                print()

            match_num += util.count_common_chunks(gold_chunks, predict_chunks)
            gold_num += len(gold_chunks)
            pred_num += len(predict_chunks)

            # fgold.write(' '.join(gold_seq) + '\n')
            # fpred.write(' '.join(predict_seq) + '\n')
            o_list = zip(input_seq, gold_seq, predict_seq)
            fout.write('\n'.join(['\t'.join(x) for x in o_list]))
            fout.write('\n\n')

            i += 1
        print()

        # fgold.close()
        # fpred.close()
        fout.close()
        if self.options.trial == 1:
            print((gold_num, pred_num, match_num))
        p, r, f1 = util.get_performance(match_num, gold_num, pred_num)
        print('P,R,F: [{0:.2f}, {1:.2f}, {2:.2f}]'.format(p * 100, r * 100, f1 * 100))
        print()

        return p, r, f1



class TransitionParser:
    def __init__(self, model : dy.ParameterCollection(), options, vocab_char : Vocab, vocab_char_count : {}, vocab_label: Vocab, vocab_label_opening: Vocab, vocab_action: Vocab, pretrain_emb : np.ndarray):
        self.vocab_char = vocab_char
        self.vocab_char_count = vocab_char_count
        self.vocab_action = vocab_action
        self.vocab_label = vocab_label          #prov
        self.vocab_label_opening = vocab_label_opening  #(prov
        self.vocab_char_count = vocab_char_count

        self.options = options
        LSTM_DIM = options.LSTM_DIM
        WORD_DIM = options.WORD_DIM
        ACTION_DIM = options.ACTION_DIM
        NUM_LAYER = options.NUM_LAYER
        NUM_ACTIONS = self.vocab_action.size()
        self.NUM_ACTIONS = NUM_ACTIONS
        NUM_WORDS = self.vocab_char.size()
        NUM_LABELS = self.vocab_label.size()

        if self.options.syntactic_composition == 2:
            self.pW_comp = model.add_parameters((LSTM_DIM, LSTM_DIM * 3))
        else:
            self.pW_comp = model.add_parameters((LSTM_DIM, LSTM_DIM * 2))
        self.pb_comp = model.add_parameters((LSTM_DIM,))

        self.pW_s2h = model.add_parameters((LSTM_DIM, LSTM_DIM * 3))
        self.pb_s2h = model.add_parameters((LSTM_DIM,))
        self.pW_act = model.add_parameters((NUM_ACTIONS, LSTM_DIM))
        self.pb_act = model.add_parameters((NUM_ACTIONS,))

        self.W_emb = model.add_parameters((LSTM_DIM, WORD_DIM))
        self.b_emb = model.add_parameters((LSTM_DIM,))

        self.S_parser_state = model.add_parameters((LSTM_DIM, LSTM_DIM))
        self.B_parser_state = model.add_parameters((LSTM_DIM, LSTM_DIM))
        self.A_parser_state = model.add_parameters((LSTM_DIM, LSTM_DIM))
        self.b_parser_state = model.add_parameters((LSTM_DIM,))

        # layers, in-dim, out-dim, model
        self.BufferRNN = dy.LSTMBuilder(NUM_LAYER, LSTM_DIM, LSTM_DIM, model)
        self.StackRNN = dy.LSTMBuilder(NUM_LAYER, LSTM_DIM, LSTM_DIM, model)
        self.ActionRNN = dy.LSTMBuilder(NUM_LAYER, ACTION_DIM, LSTM_DIM, model)
        self.CompFwdRNN = dy.LSTMBuilder(NUM_LAYER, LSTM_DIM, LSTM_DIM, model)
        self.CompBwdRNN = dy.LSTMBuilder(NUM_LAYER, LSTM_DIM, LSTM_DIM, model)

        self.BufferRNN.set_dropout(options.dropout)
        self.StackRNN.set_dropout(options.dropout)
        self.ActionRNN.set_dropout(options.dropout)
        self.CompFwdRNN.set_dropout(options.dropout)
        self.CompBwdRNN.set_dropout(options.dropout)

        self.empty_stack_emb = model.add_parameters((LSTM_DIM,))
        self.empty_buffer_emb = model.add_parameters((LSTM_DIM,))
        self.empty_actionhistory_emb = model.add_parameters((LSTM_DIM,))


        self.WORDS_LOOKUP = model.add_lookup_parameters((NUM_WORDS, WORD_DIM))  # , init=dy.NumpyInitializer(emb))
        self.ACTIONS_LOOKUP = model.add_lookup_parameters((NUM_ACTIONS, ACTION_DIM))
        self.LABEL_LOOKUP = model.add_lookup_parameters((NUM_LABELS, LSTM_DIM))
        self.LABEL_OPENING_LOOKUP = model.add_lookup_parameters((NUM_LABELS, LSTM_DIM))

        self.PRETRAIN_LOOPUP = None
        if self.options.pretrain != 'none':
            self.PRETRAIN_LOOPUP = model.add_lookup_parameters((NUM_WORDS, WORD_DIM))
            self.PRETRAIN_LOOPUP.init_from_array(pretrain_emb)
            self.W_pretrain_emb = model.add_parameters((LSTM_DIM, WORD_DIM))


    def pretrain_chunk(self, chunks):
        for label, toks in chunks:
            pass


    def decode(self, inst):
        self.predict_actions = []
        self.parse(inst, None)
        return self.predict_actions

    def IsActionForbidden_Discriminative(self, action, prev_action, buffer_size, stack_size, n):

        is_shift = (action == APPEND)
        is_reduce = (action == SEPARATE)
        is_NT = (action >= LABEL_BEGIN)

        MAX_OPEN_NTS = 100

        if is_NT and n > MAX_OPEN_NTS:
            return True

        if stack_size == 1:
            if not is_NT:
                return True
            return False

        if (IMPLICIT_REDUCE_AFTER_SHIFT):
            if is_shift and not prev_action >= LABEL_BEGIN:
                return True

        if n == 1 and buffer_size > 1:
            if IMPLICIT_REDUCE_AFTER_SHIFT and is_shift:
                return True

            if is_reduce:
                return True

        if is_reduce and prev_action >= LABEL_BEGIN:
            return True

        if is_NT and buffer_size == 1:
            return True

        if is_shift and buffer_size == 1:
            return True

        if is_reduce and stack_size < 3:
            return True

        return False

    # returns an expression of the loss for the sequence of actions
    # (that is, the oracle_actions if present or the predicted sequence otherwise)
    def parse(self, inst, oracle_actions=None):
        dy.renew_cg()
        if oracle_actions:
            oracle_actions = list(oracle_actions)
            oracle_actions.reverse()

        Stack_top = self.StackRNN.initial_state()

        ####

        t = list(inst["char"])
        if self.options.singleton == 1:
            for i in range(0, len(t)):
                if (oracle_actions != None and self.vocab_char_count[t[i]] == 1 and random.random() > 0.5):
                    t[i] = Vocab.UNK

        # if oracle_actions == None:
        #     t = list(inst["unk"])

        toks = list(t)
        toks.reverse()

        stack = []

        B_cur = self.BufferRNN.initial_state()
        buffer = []

        A = []
        Ahistory_top = self.ActionRNN.initial_state()

        empty_stack_emb = dy.parameter(self.empty_stack_emb)

        W_comp = dy.parameter(self.pW_comp)
        b_comp = dy.parameter(self.pb_comp)
        # W_s2h = dy.parameter(self.pW_s2h)
        # b_s2h = dy.parameter(self.pb_s2h)
        W_act = dy.parameter(self.pW_act)
        b_act = dy.parameter(self.pb_act)

        W_emb = dy.parameter(self.W_emb)
        b_emb = dy.parameter(self.b_emb)

        S_parser_state = dy.parameter(self.S_parser_state)
        B_parser_state = dy.parameter(self.B_parser_state)
        A_parser_state = dy.parameter(self.A_parser_state)
        b_parser_state = dy.parameter(self.b_parser_state)

        if self.PRETRAIN_LOOPUP:
            W_pretrain_emb = dy.parameter(self.W_pretrain_emb)

        losses = []

        tok_reprs = []

        prev_action = -1


        apply_dropout = self.options.dropout > 0 and oracle_actions != None

        for tok in toks:
            tok_embedding = self.WORDS_LOOKUP[tok]

            args = [b_emb, W_emb, tok_embedding]

            if self.PRETRAIN_LOOPUP:
                tok_pretrain_emb = self.PRETRAIN_LOOPUP[tok]
                args = args + [W_pretrain_emb, tok_pretrain_emb]

            tok_repr = dy.rectify(dy.affine_transform(args))
            B_cur = B_cur.add_input(tok_repr)
            buffer.append((B_cur.output(), tok_repr, self.vocab_char.i2w[tok]))
            tok_reprs.append(tok_repr)


        n = 0
        num_appended_token = 0
        while not(len(buffer) == 0 and n == 0) :
            # based on parser state, get valid actions
            valid_actions = []

            if len(buffer) > 0 and n == 1:
                valid_actions += [APPEND]

            if n == 1 and len(stack) > 0 and num_appended_token > 0:
                valid_actions += [SEPARATE]

            if len(buffer) > 0 and n == 0:
                for LABEL_ACTION in range(2, self.NUM_ACTIONS):
                     valid_actions += [LABEL_ACTION]


            # compute probability of each of the actions and choose an action
            # either from the oracle or if there is no oracle, based on the model
            action = valid_actions[0]
            log_probs = None
            if len(valid_actions) > 0:
                buffer_embedding = buffer[-1][0] if buffer else empty_stack_emb
                stack_embedding = stack[-1][0].output() if stack else empty_stack_emb  # stack[-1][0].output()  # the stack has something here
                A_embedding = A[-1][0].output() if A else empty_stack_emb

                # if apply_dropout:
                #     buffer_embedding = dy.dropout(buffer_embedding, self.options.dropout)
                #     stack_embedding = dy.dropout(stack_embedding, self.options.dropout)
                #     A_embedding = dy.dropout(A_embedding, self.options.dropout)


                h = dy.rectify(dy.affine_transform(
                    [b_parser_state, B_parser_state, buffer_embedding, S_parser_state, stack_embedding, A_parser_state,
                     A_embedding]))

                logits = dy.rectify(dy.affine_transform([b_act, W_act, h]))
                log_probs = dy.log_softmax(logits, valid_actions)
                if oracle_actions is None:
                    action = max(enumerate(log_probs.vec_value()), key=itemgetter(1))[0]
            if oracle_actions is not None:
                action = oracle_actions.pop()
                if log_probs is not None:
                    # append the action-specific loss
                    losses.append(dy.pick(log_probs, action))

            # add the current action into A Stack
            action_tok = self.vocab_action.i2w[action]
            action_emb = self.ACTIONS_LOOKUP[action]
            Astack_state, _ = A[-1] if A else (Ahistory_top, '<TOP>')
            Astack_state = Astack_state.add_input(action_emb)
            A.append((Astack_state, action_tok))

            prev_action = action

            # execute the action to update the parser state
            if action == APPEND:
                _, tok_embedding, token = buffer.pop()
                stack_state, _, _ = stack[-1] if stack else (Stack_top, '<TOP>', Stack_top)
                stack_state = stack_state.add_input(tok_embedding)
                stack.append((stack_state, token, tok_embedding))
                num_appended_token += 1

            elif action >= LABEL_BEGIN:  # LABEL(...

                label_open = action_tok[5:-1]
                label_open_id = self.vocab_label_opening.w2i[label_open]
                label_open_embedding = self.LABEL_OPENING_LOOKUP[label_open_id]

                stack_state, _, _ = stack[-1] if stack else (Stack_top, '<TOP>', Stack_top)
                stack_state = stack_state.add_input(label_open_embedding)
                stack.append((stack_state, label_open, label_open_embedding))

                n += 1
                num_appended_token = 0

            elif action == SEPARATE:  # SEP

                reduce_list = []

                while len(stack) > 0 and not stack[-1][1].startswith('('):
                    reduce_list.append(stack.pop())
                reduce_list.reverse()
                constituent = stack.pop()[1]
                constituent_tok = constituent[1:]
                constituent_tok_id = self.vocab_label.w2i[constituent_tok]
                consituent_emb = self.LABEL_LOOKUP[constituent_tok_id]



                bwd_list = [emb for state, tok, emb in reduce_list]
                fwd_list = bwd_list[::-1]

                if self.options.syntactic_composition == 1:
                    fwd_list = [consituent_emb] + fwd_list
                    bwd_list = [consituent_emb] + bwd_list

                fw_init = self.CompFwdRNN.initial_state()
                bw_init = self.CompBwdRNN.initial_state()

                fwd_exp = fw_init.transduce(fwd_list)
                bwd_exp = bw_init.transduce(bwd_list)



                if self.options.syntactic_composition == 2:
                    bi = dy.concatenate([fwd_exp[-1], bwd_exp[-1], consituent_emb])
                else:
                    bi = dy.concatenate([fwd_exp[-1], bwd_exp[-1]])


                composed_rep = dy.rectify(dy.affine_transform([b_comp, W_comp, bi]))

                top_stack_state, _, _ = stack[-1] if stack else (Stack_top, '<TOP>', Stack_top)
                top_stack_state = top_stack_state.add_input(composed_rep)
                stack.append((top_stack_state, constituent_tok + '-' + ''.join([x for _,x,_ in reduce_list]), composed_rep))

                n -= 1
                num_appended_token = 0



        if oracle_actions is None:
            self.predict_actions = [action_tok for _, action_tok in A]

        return -dy.esum(losses) if losses else None

    def actions2seq(self, actions):
        ret = []
        label_str = None
        for action_tok in actions:
            if action_tok.startswith('LABEL('):
                label_str = action_tok[6:-1]
                prefix = 'B'
            elif action_tok == 'APPEND':
                ret.append(prefix + '-' + label_str)
                prefix = 'I'
            else:  # SEP
                label_str = None

        return ret

    def predict(self, insts, fname='test'):
        total = []

        # gold_path = fname + '.gold.txt'
        # pred_path = fname + '.pred.txt'
        # fgold = open(gold_path, 'w', encoding='utf-8')
        # fpred = open(pred_path, 'w', encoding='utf-8')
        fout = open(fname, 'w', encoding='utf-8')

        match_num = 0
        gold_num = 0
        pred_num = 0

        i = 0
        for inst in insts:

            if i % 10 == 0:
                print('.', end='', flush=True)

            gold_actions = [self.vocab_action.i2w[x] for x in inst["action"]]
            predict_actions = self.decode(inst)

            if self.options.trial == 1:
                print(inst["raw_char"])
                print(inst["raw_char_normalized"])
            # print(gold_actions)
            # print(predict_actions)
            # print()

            input_seq = tuple(inst["raw_char"])
            gold_seq = inst["raw_label"]
            predict_seq = self.actions2seq(predict_actions)


            # print(gold_seq)
            # print(predict_seq)
            # print()

            gold_chunks = util.seq2chunk(gold_seq)
            predict_chunks = util.seq2chunk(predict_seq)

            if self.options.trial == 1:
                print(gold_chunks)
                print(predict_chunks)
                print()
                print()

            match_num += util.count_common_chunks(gold_chunks, predict_chunks)
            gold_num += len(gold_chunks)
            pred_num += len(predict_chunks)


            # fgold.write(' '.join(gold_seq) + '\n')
            # fpred.write(' '.join(predict_seq) + '\n')
            o_list = zip(input_seq, gold_seq, predict_seq)
            fout.write('\n'.join(['\t'.join(x) for x in o_list]))
            fout.write('\n\n')

            i += 1
        print()

        # fgold.close()
        # fpred.close()
        fout.close()

        p, r, f1 = util.get_performance(match_num, gold_num, pred_num)
        print('P,R,F: [{0:.2f}, {1:.2f}, {2:.2f}]'.format(p * 100, r * 100, f1*100))
        print()

        return p, r, f1


class TransitionParser2:
    def __init__(self, model : dy.ParameterCollection(), options, vocab_char : Vocab, vocab_char_count : {}, vocab_label: Vocab, vocab_label_opening: Vocab, vocab_action: Vocab, pretrain_emb : np.ndarray):
        self.APPEND = 0
        self.LABEL_BEGIN = 1


        self.vocab_char = vocab_char
        self.vocab_char_count = vocab_char_count
        self.vocab_action = vocab_action
        self.vocab_label = vocab_label          #prov
        self.vocab_label_opening = vocab_label_opening  #(prov
        self.vocab_char_count = vocab_char_count

        self.options = options
        LSTM_DIM = options.LSTM_DIM
        WORD_DIM = options.WORD_DIM
        ACTION_DIM = options.ACTION_DIM
        NUM_LAYER = options.NUM_LAYER
        NUM_ACTIONS = self.vocab_action.size()
        self.NUM_ACTIONS = NUM_ACTIONS
        NUM_WORDS = self.vocab_char.size()
        NUM_LABELS = self.vocab_label.size()

        if self.options.syntactic_composition == 2:
            self.pW_comp = model.add_parameters((LSTM_DIM, LSTM_DIM * 3))
        else:
            self.pW_comp = model.add_parameters((LSTM_DIM, LSTM_DIM * 2))

        self.pb_comp = model.add_parameters((LSTM_DIM,))
        self.pW_s2h = model.add_parameters((LSTM_DIM, LSTM_DIM * 3))
        self.pb_s2h = model.add_parameters((LSTM_DIM,))
        self.pW_act = model.add_parameters((NUM_ACTIONS, LSTM_DIM))
        self.pb_act = model.add_parameters((NUM_ACTIONS,))

        self.W_emb = model.add_parameters((LSTM_DIM, WORD_DIM))
        self.b_emb = model.add_parameters((LSTM_DIM,))

        self.S_parser_state = model.add_parameters((LSTM_DIM, LSTM_DIM))
        self.B_parser_state = model.add_parameters((LSTM_DIM, LSTM_DIM))
        self.A_parser_state = model.add_parameters((LSTM_DIM, LSTM_DIM))
        self.b_parser_state = model.add_parameters((LSTM_DIM,))

        # layers, in-dim, out-dim, model
        self.BufferRNN = dy.LSTMBuilder(NUM_LAYER, LSTM_DIM, LSTM_DIM, model)
        self.StackRNN = dy.LSTMBuilder(NUM_LAYER, LSTM_DIM, LSTM_DIM, model)
        self.ActionRNN = dy.LSTMBuilder(NUM_LAYER, ACTION_DIM, LSTM_DIM, model)
        self.CompFwdRNN = dy.LSTMBuilder(NUM_LAYER, LSTM_DIM, LSTM_DIM, model)
        self.CompBwdRNN = dy.LSTMBuilder(NUM_LAYER, LSTM_DIM, LSTM_DIM, model)

        self.BufferRNN.set_dropout(options.dropout)
        self.StackRNN.set_dropout(options.dropout)
        self.ActionRNN.set_dropout(options.dropout)
        self.CompFwdRNN.set_dropout(options.dropout)
        self.CompBwdRNN.set_dropout(options.dropout)

        self.empty_stack_emb = model.add_parameters((LSTM_DIM,))
        self.empty_buffer_emb = model.add_parameters((LSTM_DIM,))
        self.empty_actionhistory_emb = model.add_parameters((LSTM_DIM,))


        self.WORDS_LOOKUP = model.add_lookup_parameters((NUM_WORDS, WORD_DIM))  # , init=dy.NumpyInitializer(emb))
        self.ACTIONS_LOOKUP = model.add_lookup_parameters((NUM_ACTIONS, ACTION_DIM))
        self.LABEL_LOOKUP = model.add_lookup_parameters((NUM_LABELS, LSTM_DIM))
        self.LABEL_OPENING_LOOKUP = model.add_lookup_parameters((NUM_LABELS, LSTM_DIM))

        self.PRETRAIN_LOOPUP = None
        if self.options.pretrain != 'none':
            self.PRETRAIN_LOOPUP = model.add_lookup_parameters((NUM_WORDS, WORD_DIM))
            self.PRETRAIN_LOOPUP.init_from_array(pretrain_emb)
            self.W_pretrain_emb = model.add_parameters((LSTM_DIM, WORD_DIM))


    def pretrain_chunk(self, chunks):
        for label, toks in chunks:
            pass


    def decode(self, inst):
        self.predict_actions = []
        self.parse(inst, None)
        return self.predict_actions


    # returns an expression of the loss for the sequence of actions
    # (that is, the oracle_actions if present or the predicted sequence otherwise)
    def parse(self, inst, oracle_actions=None):
        dy.renew_cg()
        if oracle_actions:
            oracle_actions = list(oracle_actions)
            oracle_actions.reverse()

        Stack_top = self.StackRNN.initial_state()

        ####

        t = list(inst["char"])
        for i in range(0, len(t)):
            if (oracle_actions != None and self.vocab_char_count[t[i]] == 1 and random.random() > 0.5):
                t[i] = Vocab.UNK

        # if oracle_actions == None:
        #     t = list(inst["unk"])

        toks = list(t)
        toks.reverse()

        stack = []

        B_cur = self.BufferRNN.initial_state()
        buffer = []

        A = []
        Ahistory_top = self.ActionRNN.initial_state()

        empty_stack_emb = dy.parameter(self.empty_stack_emb)

        W_comp = dy.parameter(self.pW_comp)
        b_comp = dy.parameter(self.pb_comp)
        # W_s2h = dy.parameter(self.pW_s2h)
        # b_s2h = dy.parameter(self.pb_s2h)
        W_act = dy.parameter(self.pW_act)
        b_act = dy.parameter(self.pb_act)

        W_emb = dy.parameter(self.W_emb)
        b_emb = dy.parameter(self.b_emb)

        S_parser_state = dy.parameter(self.S_parser_state)
        B_parser_state = dy.parameter(self.B_parser_state)
        A_parser_state = dy.parameter(self.A_parser_state)
        b_parser_state = dy.parameter(self.b_parser_state)

        if self.PRETRAIN_LOOPUP:
            W_pretrain_emb = dy.parameter(self.W_pretrain_emb)

        losses = []

        tok_reprs = []

        prev_action = -1


        apply_dropout = self.options.dropout > 0 and oracle_actions != None

        for tok in toks:
            tok_embedding = self.WORDS_LOOKUP[tok]

            args = [b_emb, W_emb, tok_embedding]

            if self.PRETRAIN_LOOPUP:
                tok_pretrain_emb = self.PRETRAIN_LOOPUP[tok]
                args = args + [W_pretrain_emb, tok_pretrain_emb]

            tok_repr = dy.rectify(dy.affine_transform(args))
            B_cur = B_cur.add_input(tok_repr)
            buffer.append((B_cur.output(), tok_repr, self.vocab_char.i2w[tok]))
            tok_reprs.append(tok_repr)


        n = 0
        num_appended_token = 0
        while not(len(buffer) == 0 and num_appended_token == 0) :
            # based on parser state, get valid actions
            valid_actions = []

            if len(buffer) > 0 : #and n == 1:
                valid_actions += [self.APPEND]

            # if n == 1 and len(stack) > 0 and num_appended_token > 0:
            #     valid_actions += [SEPARATE]

            if num_appended_token > 0 : #n == 0:
                for LABEL_ACTION in range(2, self.NUM_ACTIONS):
                     valid_actions += [LABEL_ACTION]


            # compute probability of each of the actions and choose an action
            # either from the oracle or if there is no oracle, based on the model
            action = valid_actions[0]
            log_probs = None
            if len(valid_actions) > 0:
                buffer_embedding = buffer[-1][0] if buffer else empty_stack_emb
                stack_embedding = stack[-1][0].output() if stack else empty_stack_emb  # stack[-1][0].output()  # the stack has something here
                A_embedding = A[-1][0].output() if A else empty_stack_emb

                # if apply_dropout:
                #     buffer_embedding = dy.dropout(buffer_embedding, self.options.dropout)
                #     stack_embedding = dy.dropout(stack_embedding, self.options.dropout)
                #     A_embedding = dy.dropout(A_embedding, self.options.dropout)


                h = dy.rectify(dy.affine_transform(
                    [b_parser_state, B_parser_state, buffer_embedding, S_parser_state, stack_embedding, A_parser_state,
                     A_embedding]))

                logits = dy.rectify(dy.affine_transform([b_act, W_act, h]))
                log_probs = dy.log_softmax(logits, valid_actions)
                if oracle_actions is None:
                    action = max(enumerate(log_probs.vec_value()), key=itemgetter(1))[0]
            if oracle_actions is not None:
                action = oracle_actions.pop()
                if log_probs is not None:
                    # append the action-specific loss
                    losses.append(dy.pick(log_probs, action))

            # add the current action into A Stack
            action_tok = self.vocab_action.i2w[action]
            action_emb = self.ACTIONS_LOOKUP[action]
            Astack_state, _ = A[-1] if A else (Ahistory_top, '<TOP>')
            Astack_state = Astack_state.add_input(action_emb)
            A.append((Astack_state, action_tok))

            prev_action = action

            # execute the action to update the parser state
            if action == self.APPEND:
                _, tok_embedding, token = buffer.pop()
                stack_state, _, _ = stack[-1] if stack else (Stack_top, '<TOP>', Stack_top)
                stack_state = stack_state.add_input(tok_embedding)
                stack.append((stack_state, token, tok_embedding))
                num_appended_token += 1


            elif action >= self.LABEL_BEGIN : #== SEPARATE:  # SEP

                reduce_list = []

                while len(stack) > 0 and not stack[-1][1].startswith('('):
                    reduce_list.append(stack.pop())
                reduce_list.reverse()
                # constituent = stack.pop()[1]
                constituent_tok = action_tok[6:-1]
                constituent_tok_id = self.vocab_label.w2i[constituent_tok]

                consituent_emb = self.LABEL_LOOKUP[constituent_tok_id]



                bwd_list = [emb for state, tok, emb in reduce_list]
                fwd_list = bwd_list[::-1]

                if self.options.syntactic_composition == 1:
                    fwd_list = [consituent_emb] + fwd_list
                    bwd_list = [consituent_emb] + bwd_list

                fw_init = self.CompFwdRNN.initial_state()
                bw_init = self.CompBwdRNN.initial_state()

                fwd_exp = fw_init.transduce(fwd_list)
                bwd_exp = bw_init.transduce(bwd_list)

                if self.options.syntactic_composition == 2:
                    bi = dy.concatenate([fwd_exp[-1], bwd_exp[-1], consituent_emb])
                else:
                    bi = dy.concatenate([fwd_exp[-1], bwd_exp[-1]])


                composed_rep = dy.rectify(dy.affine_transform([b_comp, W_comp, bi]))

                top_stack_state, _, _ = stack[-1] if stack else (Stack_top, '<TOP>', Stack_top)
                top_stack_state = top_stack_state.add_input(composed_rep)
                stack.append((top_stack_state, '('+ constituent_tok + ') -' + ''.join([x for _,x,_ in reduce_list]), composed_rep))

                n -= 1
                num_appended_token = 0



        if oracle_actions is None:
            self.predict_actions = [action_tok for _, action_tok in A]

        return -dy.esum(losses) if losses else None

    def actions2seq(self, actions):
        ret = []
        label_str = None
        for action_tok in actions:
            if action_tok.startswith('LABEL('):
                label_str = action_tok[6:-1]
                prefix = 'B'
            elif action_tok == 'APPEND':
                ret.append(prefix + '-' + label_str)
                prefix = 'I'
            else:  # SEP
                label_str = None

        return ret

    def predict(self, insts, fname='test'):
        total = []

        # gold_path = fname + '.gold.txt'
        # pred_path = fname + '.pred.txt'
        # fgold = open(gold_path, 'w', encoding='utf-8')
        # fpred = open(pred_path, 'w', encoding='utf-8')

        fout = open(fname, 'w', encoding='utf-8')

        match_num = 0
        gold_num = 0
        pred_num = 0

        i = 0
        for inst in insts:

            if i % 10 == 0:
                print('.', end='', flush=True)

            gold_actions = [self.vocab_action.i2w[x] for x in inst["action"]]
            predict_actions = self.decode(inst)

            if self.options.trial == 1:
                print(inst["raw_char"])
                print(inst["raw_char_normalized"])
            # print(gold_actions)
            # print(predict_actions)
            # print()





            # print(gold_seq)
            # print(predict_seq)
            # print()

            gold_chunks = util.action2chunk(gold_actions)
            predict_chunks = util.action2chunk(predict_actions)


            input_seq = tuple(inst["raw_char"])
            gold_seq = tuple(inst["raw_label"])
            predict_seq = tuple(util.chunk2seq(predict_chunks))

            if self.options.trial == 1:
                print(gold_chunks)
                print(predict_chunks)
                print()
                print()

            match_num += util.count_common_chunks(gold_chunks, predict_chunks)
            gold_num += len(gold_chunks)
            pred_num += len(predict_chunks)

            o_list = zip(input_seq, gold_seq, predict_seq)
            fout.write('\n'.join(['\t'.join(x) for x in o_list]))
            fout.write('\n\n')


            # fgold.write(' '.join(gold_seq) + '\n')
            # fpred.write(' '.join(predict_seq) + '\n')

            i += 1
        print()

        # fgold.close()
        # fpred.close()
        fout.close()

        p, r, f1 = util.get_performance(match_num, gold_num, pred_num)
        print('P,R,F: [{0:.2f}, {1:.2f}, {2:.2f}]'.format(p * 100, r * 100, f1*100))
        print()

        return p, r, f1







class TransitionParser_RBT(TransitionParser):
    def __init__(self, model : dy.ParameterCollection(), options, vocab_char : Vocab, vocab_char_count : {}, vocab_label: Vocab, vocab_label_opening: Vocab, vocab_action: Vocab, pretrain_emb : np.ndarray):
        self.APPEND = 0
        self.LABEL_BEGIN = 1


        self.vocab_char = vocab_char
        self.vocab_char_count = vocab_char_count
        self.vocab_action = vocab_action
        self.vocab_label = vocab_label          #prov
        self.vocab_label_opening = vocab_label_opening  #(prov
        self.vocab_char_count = vocab_char_count

        self.options = options
        LSTM_DIM = options.LSTM_DIM
        WORD_DIM = options.WORD_DIM
        ACTION_DIM = options.ACTION_DIM
        NUM_LAYER = options.NUM_LAYER
        NUM_ACTIONS = self.vocab_action.size()
        self.NUM_ACTIONS = NUM_ACTIONS
        NUM_WORDS = self.vocab_char.size()
        NUM_LABELS = self.vocab_label.size()


        self.pW_comp = model.add_parameters((LSTM_DIM, LSTM_DIM * 2))
        self.pb_comp = model.add_parameters((LSTM_DIM,))
        self.pW_s2h = model.add_parameters((LSTM_DIM, LSTM_DIM * 3))
        self.pb_s2h = model.add_parameters((LSTM_DIM,))
        self.pW_act = model.add_parameters((NUM_ACTIONS, LSTM_DIM))
        self.pb_act = model.add_parameters((NUM_ACTIONS,))

        self.W_emb = model.add_parameters((LSTM_DIM, WORD_DIM))
        self.b_emb = model.add_parameters((LSTM_DIM,))

        self.S_parser_state = model.add_parameters((LSTM_DIM, LSTM_DIM))
        self.B_parser_state = model.add_parameters((LSTM_DIM, LSTM_DIM))
        self.A_parser_state = model.add_parameters((LSTM_DIM, LSTM_DIM))
        self.b_parser_state = model.add_parameters((LSTM_DIM,))

        # layers, in-dim, out-dim, model
        self.BufferRNN = dy.LSTMBuilder(NUM_LAYER, LSTM_DIM, LSTM_DIM, model)
        self.StackRNN = dy.LSTMBuilder(NUM_LAYER, LSTM_DIM, LSTM_DIM, model)
        self.ActionRNN = dy.LSTMBuilder(NUM_LAYER, ACTION_DIM, LSTM_DIM, model)
        self.CompFwdRNN = dy.LSTMBuilder(NUM_LAYER, LSTM_DIM, LSTM_DIM, model)
        self.CompBwdRNN = dy.LSTMBuilder(NUM_LAYER, LSTM_DIM, LSTM_DIM, model)

        self.BufferRNN.set_dropout(options.dropout)
        self.StackRNN.set_dropout(options.dropout)
        self.ActionRNN.set_dropout(options.dropout)
        self.CompFwdRNN.set_dropout(options.dropout)
        self.CompBwdRNN.set_dropout(options.dropout)

        self.empty_stack_emb = model.add_parameters((LSTM_DIM,))
        self.empty_buffer_emb = model.add_parameters((LSTM_DIM,))
        self.empty_actionhistory_emb = model.add_parameters((LSTM_DIM,))


        self.WORDS_LOOKUP = model.add_lookup_parameters((NUM_WORDS, WORD_DIM))  # , init=dy.NumpyInitializer(emb))
        self.ACTIONS_LOOKUP = model.add_lookup_parameters((NUM_ACTIONS, ACTION_DIM))
        self.LABEL_LOOKUP = model.add_lookup_parameters((NUM_LABELS, LSTM_DIM))
        self.LABEL_OPENING_LOOKUP = model.add_lookup_parameters((NUM_LABELS, LSTM_DIM))

        self.PRETRAIN_LOOPUP = None
        if self.options.pretrain != 'none':
            self.PRETRAIN_LOOPUP = model.add_lookup_parameters((NUM_WORDS, WORD_DIM))
            self.PRETRAIN_LOOPUP.init_from_array(pretrain_emb)
            self.W_pretrain_emb = model.add_parameters((LSTM_DIM, WORD_DIM))




    def decode(self, inst):
        self.predict_actions = []
        self.parse(inst, None)
        return self.predict_actions



    # returns an expression of the loss for the sequence of actions
    # (that is, the oracle_actions if present or the predicted sequence otherwise)
    def parse(self, inst, oracle_actions=None):
        dy.renew_cg()
        if oracle_actions:
            oracle_actions = list(oracle_actions)
            oracle_actions.reverse()

        Stack_top = self.StackRNN.initial_state()

        ####

        t = list(inst["char"])
        if self.options.singleton == 1:
            for i in range(0, len(t)):
                if (oracle_actions != None and self.vocab_char_count[t[i]] == 1 and random.random() > 0.5):
                    t[i] = Vocab.UNK

        # if oracle_actions == None:
        #     t = list(inst["unk"])

        toks = list(t)
        toks.reverse()

        stack = []

        B_cur = self.BufferRNN.initial_state()
        buffer = []

        A = []
        Ahistory_top = self.ActionRNN.initial_state()

        empty_stack_emb = dy.parameter(self.empty_stack_emb)

        W_comp = dy.parameter(self.pW_comp)
        b_comp = dy.parameter(self.pb_comp)
        # W_s2h = dy.parameter(self.pW_s2h)
        # b_s2h = dy.parameter(self.pb_s2h)
        W_act = dy.parameter(self.pW_act)
        b_act = dy.parameter(self.pb_act)

        W_emb = dy.parameter(self.W_emb)
        b_emb = dy.parameter(self.b_emb)

        S_parser_state = dy.parameter(self.S_parser_state)
        B_parser_state = dy.parameter(self.B_parser_state)
        A_parser_state = dy.parameter(self.A_parser_state)
        b_parser_state = dy.parameter(self.b_parser_state)

        if self.PRETRAIN_LOOPUP:
            W_pretrain_emb = dy.parameter(self.W_pretrain_emb)

        losses = []

        tok_reprs = []

        prev_action = -1


        apply_dropout = self.options.dropout > 0 and oracle_actions != None

        for tok in toks:
            tok_embedding = self.WORDS_LOOKUP[tok]

            args = [b_emb, W_emb, tok_embedding]

            if self.PRETRAIN_LOOPUP:
                tok_pretrain_emb = self.PRETRAIN_LOOPUP[tok]
                args = args + [W_pretrain_emb, tok_pretrain_emb]

            tok_repr = dy.rectify(dy.affine_transform(args))
            B_cur = B_cur.add_input(tok_repr)
            buffer.append((B_cur.output(), tok_repr, self.vocab_char.i2w[tok]))
            tok_reprs.append(tok_repr)


        n = 0
        num_appended_token = []
        opening_nonterminal = []
        while not(len(buffer) == 0 and n == 0) :
            # based on parser state, get valid actions
            valid_actions = []

            if len(buffer) > 0 and n >= 1 and len(opening_nonterminal) > 0 and not opening_nonterminal[-1].endswith("'") :
                valid_actions += [APPEND]

            if n >= 1 and len(stack) > 0 and len(num_appended_token) > 0 and num_appended_token[-1] > 0:
                valid_actions += [SEPARATE]

            if len(buffer) > 0 and n < 100: # and (len(num_appended_token) >= 1 or num_appended_token[-1] > 0):
                for LABEL_ACTION in range(2, self.NUM_ACTIONS):
                    if len(opening_nonterminal) > 0 and self.vocab_action.i2w[LABEL_ACTION][5:-1] == opening_nonterminal[-1]:
                        pass
                    else:
                        valid_actions += [LABEL_ACTION]


            # compute probability of each of the actions and choose an action
            # either from the oracle or if there is no oracle, based on the model
            try:
                action = valid_actions[0]
            except:
                print()
                exit()


            log_probs = None
            if len(valid_actions) > 0:
                buffer_embedding = buffer[-1][0] if buffer else empty_stack_emb
                stack_embedding = stack[-1][0].output() if stack else empty_stack_emb  # stack[-1][0].output()  # the stack has something here
                A_embedding = A[-1][0].output() if A else empty_stack_emb

                # if apply_dropout:
                #     buffer_embedding = dy.dropout(buffer_embedding, self.options.dropout)
                #     stack_embedding = dy.dropout(stack_embedding, self.options.dropout)
                #     A_embedding = dy.dropout(A_embedding, self.options.dropout)


                h = dy.rectify(dy.affine_transform(
                    [b_parser_state, B_parser_state, buffer_embedding, S_parser_state, stack_embedding, A_parser_state,
                     A_embedding]))

                logits = dy.rectify(dy.affine_transform([b_act, W_act, h]))
                log_probs = dy.log_softmax(logits, valid_actions)
                if oracle_actions is None:
                    action = max(enumerate(log_probs.vec_value()), key=itemgetter(1))[0]
            if oracle_actions is not None:
                action = oracle_actions.pop()
                if log_probs is not None:
                    # append the action-specific loss
                    losses.append(dy.pick(log_probs, action))

            # add the current action into A Stack
            action_tok = self.vocab_action.i2w[action]
            action_emb = self.ACTIONS_LOOKUP[action]
            Astack_state, _ = A[-1] if A else (Ahistory_top, '<TOP>')
            Astack_state = Astack_state.add_input(action_emb)
            A.append((Astack_state, action_tok))

            prev_action = action

            # execute the action to update the parser state
            if action == APPEND:
                _, tok_embedding, token = buffer.pop()
                stack_state, _, _ = stack[-1] if stack else (Stack_top, '<TOP>', Stack_top)
                stack_state = stack_state.add_input(tok_embedding)
                stack.append((stack_state, token, tok_embedding))
                num_appended_token[-1] += 1

            elif action >= LABEL_BEGIN:  # LABEL(...

                label_open = action_tok[5:-1]
                label_open_id = self.vocab_label_opening.w2i[label_open]
                label_open_embedding = self.LABEL_OPENING_LOOKUP[label_open_id]

                stack_state, _, _ = stack[-1] if stack else (Stack_top, '<TOP>', Stack_top)
                stack_state = stack_state.add_input(label_open_embedding)
                stack.append((stack_state, label_open, label_open_embedding))

                n += 1
                num_appended_token.append(0)
                opening_nonterminal.append(label_open)

            elif action == SEPARATE:  # SEP

                reduce_list = []

                while len(stack) > 0 and not stack[-1][1].startswith('('):
                    reduce_list.append(stack.pop())
                reduce_list.reverse()
                constituent = stack.pop()[1]
                constituent_tok = constituent[1:]
                constituent_tok_id = self.vocab_label.w2i[constituent_tok]
                consituent_emb = self.LABEL_LOOKUP[constituent_tok_id]



                bwd_list = [emb for state, tok, emb in reduce_list]
                fwd_list = bwd_list[::-1]

                if self.options.syntactic_composition == 1:
                    fwd_list = [consituent_emb] + fwd_list
                    bwd_list = [consituent_emb] + bwd_list

                fw_init = self.CompFwdRNN.initial_state()
                bw_init = self.CompBwdRNN.initial_state()

                fwd_exp = fw_init.transduce(fwd_list)
                bwd_exp = bw_init.transduce(bwd_list)

                bi = dy.concatenate([fwd_exp[-1], consituent_emb]) # bwd_exp[-1],


                composed_rep = dy.rectify(dy.affine_transform([b_comp, W_comp, bi]))

                top_stack_state, _, _ = stack[-1] if stack else (Stack_top, '<TOP>', Stack_top)
                top_stack_state = top_stack_state.add_input(composed_rep)
                stack.append((top_stack_state, constituent_tok + '-' + ''.join([x for _,x,_ in reduce_list]), composed_rep))

                n -= 1
                num_appended_token.pop()
                if len(num_appended_token) > 0 :
                    num_appended_token[-1] += 1

                opening_nonterminal.pop()



        if oracle_actions is None:
            self.predict_actions = [action_tok for _, action_tok in A]

        return -dy.esum(losses) if losses else None



    def predict(self, insts, fname='test'):


        fout = open(fname, 'w', encoding='utf-8')
        fout_action = open(fname + '.action.txt', 'w', encoding='utf-8')

        match_num = 0
        gold_num = 0
        pred_num = 0

        i = 0
        for inst in insts:

            if i % 10 == 0:
                print('.', end='', flush=True)

            gold_actions = [self.vocab_action.i2w[x] for x in inst["action"]]
            predict_actions = self.decode(inst)

            # fout_action.write(','.join(gold_actions) + '\n')
            # fout_action.write(','.join(predict_actions) + '\n\n')

            fout_action.write(util.action2treestr(gold_actions, inst["raw_char"]) + "\n")
            fout_action.write(util.action2treestr(predict_actions, inst["raw_char"]) + "\n\n")

            if self.options.trial == 1:
                print(inst["raw_char"])
                print(inst["raw_char_normalized"])

                print(colored(util.action2treestr(gold_actions, inst["raw_char"]), 'red'))
                print(colored(util.action2treestr(predict_actions, inst["raw_char"]), 'red'))
            # print(gold_actions)
            # print(predict_actions)
            # print()

            # print(gold_seq)
            # print(predict_seq)
            # print()

            gold_chunks = util.action2chunk_rbt(gold_actions)
            predict_chunks = util.action2chunk_rbt(predict_actions)

            input_seq = tuple(inst["raw_char"])
            gold_seq = tuple(inst["raw_label"])
            predict_seq = tuple(util.chunk2seq(predict_chunks))

            if self.options.trial == 1:
                print(colored(str(gold_actions), 'yellow'))
                print(colored(str(predict_actions), 'yellow'))
                print(gold_chunks)
                print(predict_chunks)
                print()
                print()

            match_num += util.count_common_chunks(gold_chunks, predict_chunks)
            gold_num += len(gold_chunks)
            pred_num += len(predict_chunks)

            o_list = zip(input_seq, gold_seq, predict_seq)
            fout.write('\n'.join(['\t'.join(x) for x in o_list]))
            fout.write('\n\n')



            i += 1
        print()


        fout.close()
        fout_action.close()

        p, r, f1 = util.get_performance(match_num, gold_num, pred_num)
        print('P,R,F: [{0:.2f}, {1:.2f}, {2:.2f}]'.format(p * 100, r * 100, f1 * 100))
        print()

        return p, r, f1
