import random
import os
import util
from optparse import OptionParser


usage = "ttest.py --file1 [file1] --file2 [file2]"

parser = OptionParser(usage=usage)
parser.add_option("--file1", type="string", help="file1", default="", dest="file1")
parser.add_option("--file2", type="string", help="file2", default="", dest="file2")
#parser.add_option("--mode", type="string", help="mode", default="scope", dest="mode")
(options, args) = parser.parse_args()

class TTest:
    """TTest"""

    def __init__(self, nSample = 10000, debug = False):
        self.nSample = nSample
        random.seed(99997)
        self.debug = debug

    def loadfile(self, filename):
        f = open(filename, 'r', encoding='utf-8')
        insts = []
        gold = []
        pred = []
        for line in f:
            line = line.strip()
            if len(line) == 0:
                if (len(gold) > 0):
                    insts.append((gold, pred))
                gold = list()
                pred = list()
            else:
                fields = line.split('\t')
                gold.append(fields[1])
                pred.append(fields[2])

        f.close()
        return  insts

    def loadfiles(self, filename1, filename2):
        self.inst1 = self.loadfile(filename1)
        print(self.eval(self.inst1))
        self.inst2 = self.loadfile(filename2)
        print(self.eval(self.inst2))


    def ttest(self):

        insts1, insts2 = self.inst1, self.inst2

        insts1 = insts1[:-1]

        count = 0

        # print('length:')
        # print(len(insts1), len(insts2))
        assert(len(insts1) == len(insts2))

        K = len(insts1)
        print('K:', K)

        for i in range(0, self.nSample):

            if i % 1000 == 0:
                print('.', end='', flush=True)

            selectInst1 = []
            selectInst2 = []

            for k in range(0, K):
                idx = random.randrange(K)
                selectInst1.append(insts1[idx])
                selectInst2.append(insts2[idx])

            score1 = self.eval(selectInst1)
            score2 = self.eval(selectInst2)

            if (score1 > score2):
                count += 1


        p = (self.nSample + 0.0 - count) / self.nSample

        print()

        return p


    """inst contains sequence of (gold, pred)"""
    def eval(self,insts):


        Ret = (0.0, 0.0, 0.0)

        for inst in insts:
            ret = self.evalInst(inst)

            Ret = [sum(x) for x in zip(*(Ret, ret))]
        if self.debug:
            print('Ret:', Ret)
        gold, pred, match = Ret


        P = (match + 0.0) / pred
        R = (match + 0.0) / gold
        F1 = 2.0 * P * R / (P + R) if abs(P + R) > 1e-5 else 0.0

        return F1

    def evalInst(self, inst):
        return self.evalchunks(inst)


    def evalchunks(self, inst):
        gold, pred = inst
        gold_chunks = util.seq2chunk(gold)
        pred_chunks = util.seq2chunk(pred)

        return len(gold_chunks), len(pred_chunks), util.count_common_chunks(gold_chunks, pred_chunks)


def main():
    ttest = TTest()
    ttest.loadfiles(options.file1, options.file2)
    p = ttest.ttest()
    print('p=',p)

if __name__ == "__main__":
    main()


