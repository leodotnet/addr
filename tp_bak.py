from operator import itemgetter
from itertools import count
from collections import Counter, defaultdict
import random
import dynet as dy
import numpy as np
import re
from reader import Vocab
import util

IMPLICIT_REDUCE_AFTER_SHIFT = 0

APPEND = 0  #APP
SEPARATE = 1 #SEP
LABEL_BEGIN = 2  #LBL

LABEL_LIMIT = 100


class BiLSTM_CRF:

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

        # LSTM parameters
        self.bi_lstm = dy.BiRNNBuilder(NUM_LAYER, WORD_DIM, LSTM_DIM, self.model, dy.LSTMBuilder)

        # Matrix that maps from Bi-LSTM output to num tags
        self.lstm_to_tags_params = self.model.add_parameters((NUM_LABELS, LSTM_DIM))
        self.lstm_to_tags_bias = self.model.add_parameters(NUM_LABELS)
        self.mlp_out = self.model.add_parameters((NUM_LABELS, NUM_LABELS))
        self.mlp_out_bias = self.model.add_parameters(NUM_LABELS)

        # Transition matrix for tagging layer, [i,j] is score of transitioning to i from j
        self.transitions = self.model.add_lookup_parameters((NUM_LABELS, NUM_LABELS))

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
        scores = []
        for rep in lstm_out:
            score_t = O * dy.tanh(H * rep + Hb) + Ob
            scores.append(score_t)

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

    def neg_log_loss(self, sentence, tags):
        observations = self.build_tagging_graph(sentence)
        gold_score = self.score_sentence(observations, tags)
        forward_score = self.forward(observations)
        return forward_score - gold_score

    def parse(self, observations):

        def log_sum_exp(scores):
            npval = scores.npvalue()
            argmax_score = np.argmax(npval)
            max_score_expr = dy.pick(scores, argmax_score)
            max_score_expr_broadcast = dy.concatenate([max_score_expr] * self.tagset_size)
            return max_score_expr + dy.log(dy.sum_cols(dy.transpose(dy.exp(scores - max_score_expr_broadcast))))

        init_alphas = [-1e10] * self.tagset_size
        init_alphas[self.vocab_label.w2i["<START>"]] = 0
        for_expr = dy.inputVector(init_alphas)
        for obs in observations:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                obs_broadcast = dy.concatenate([dy.pick(obs, next_tag)] * self.tagset_size)
                next_tag_expr = for_expr + self.transitions[next_tag] + obs_broadcast
                alphas_t.append(log_sum_exp(next_tag_expr))
            for_expr = dy.concatenate(alphas_t)
        terminal_expr = for_expr + self.transitions[self.vocab_label.w2i["<STOP>"]]
        alpha = log_sum_exp(terminal_expr)
        return alpha

    def viterbi_decoding(self, observations):
        backpointers = []
        init_vvars = [-1e10] * self.tagset_size
        init_vvars[self.vocab_label.w2i["<START>"]] = 0  # <Start> has all the probability
        for_expr = dy.inputVector(init_vvars)
        trans_exprs = [self.transitions[idx] for idx in range(self.tagset_size)]
        for obs in observations:
            bptrs_t = []
            vvars_t = []
            for next_tag in range(self.tagset_size):
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

    @property
    def model(self):
        return self.model


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
            fout.write('\n')

            i += 1
        print()

        # fgold.close()
        # fpred.close()
        fout.close()

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

                fwd_list = [consituent_emb] + fwd_list
                bwd_list = [consituent_emb] + bwd_list

                fw_init = self.CompFwdRNN.initial_state()
                bw_init = self.CompBwdRNN.initial_state()

                fwd_exp = fw_init.transduce(fwd_list)
                bwd_exp = bw_init.transduce(bwd_list)

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

        gold_path = fname + '.gold.txt'
        pred_path = fname + '.pred.txt'
        fgold = open(gold_path, 'w', encoding='utf-8')
        fpred = open(pred_path, 'w', encoding='utf-8')

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


            fgold.write(' '.join(gold_seq) + '\n')
            fpred.write(' '.join(predict_seq) + '\n')

            i += 1
        print()

        fgold.close()
        fpred.close()

        p, r, f1 = util.get_performance(match_num, gold_num, pred_num)
        print('P,R,F: [{0:.2f}, {1:.2f}, {2:.2f}]'.format(p * 100, r * 100, f1*100))
        print()

        return p, r, f1


class TransitionParser2:
    def __init__(self, model : dy.ParameterCollection(), options, vocab_char : Vocab, vocab_char_count : {}, vocab_label: Vocab, vocab_label_opening: Vocab, vocab_action: Vocab, pretrain_emb : np.ndarray):
        self.APPEND == 0
        self.LABEL_BEGIN == 1


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

            if len(buffer) >= 0 and num_appended_token > 0 : #n == 0:
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

            # elif action >= LABEL_BEGIN:  # LABEL(...
            #
            #     label_open = action_tok[5:-1]
            #     label_open_id = self.vocab_label_opening.w2i[label_open]
            #     label_open_embedding = self.LABEL_OPENING_LOOKUP[label_open_id]
            #
            #     stack_state, _, _ = stack[-1] if stack else (Stack_top, '<TOP>', Stack_top)
            #     stack_state = stack_state.add_input(label_open_embedding)
            #     stack.append((stack_state, label_open, label_open_embedding))
            #
            #     n += 1
            #     num_appended_token = 0

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

                fwd_list = [consituent_emb] + fwd_list
                bwd_list = [consituent_emb] + bwd_list

                fw_init = self.CompFwdRNN.initial_state()
                bw_init = self.CompBwdRNN.initial_state()

                fwd_exp = fw_init.transduce(fwd_list)
                bwd_exp = bw_init.transduce(bwd_list)

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
            fout.write('\n')


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
