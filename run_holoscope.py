import sys
from gendenseblock import *
import numpy as np
import io as basicio
from holoscopeFraudDect import *
from mytools.ioutil import loadSimpleList
from sklearn.metrics import roc_auc_score

def precision_recall_sets(trueset, predset):
    correct = 0
    for p in predset:
        if p in trueset: correct += 1
    precision = 0 if len(predset) == 0 else float(correct)/len(predset)
    recall = 0 if len(trueset)==0 else float(correct)/len(trueset)
    return precision, recall

def Fmeasure(pre, rec):
    if pre+rec > 0:
	F = 2*pre*rec/(pre+rec)
    else:
	F = 0
    return F

def Fmeasures(pres, recs):
    Fs = 2*np.multiply(pres, recs)
    possidx = Fs>0
    Fs[possidx] = np.divide(Fs[possidx], (pres+recs)[possidx])
    return Fs

def tpr_fpr_sets(trueset, predset):
    'true positive and false negative'
    tp, fp = 0, 0
    for p in predset:
        if p in trueset:
            tp += 1
    tpr = float(tp)/len(predset)
    return tpr, 1-tpr

def auc_trueset_rankscore(ts, rscore):
    nV = len(rscore)
    truelabel = np.zeros(nV, dtype=int)
    truelabel[list(ts)]=1
    return roc_auc_score(truelabel, rscore)

def auc_trueset_ranklist(ts, rlist):
    rscore = np.array(rlist[::-1]).argsort()
    nV = len(rscore)
    truelabel = np.zeros(nV, dtype=int)
    truelabel[list(ts)]=1
    return roc_auc_score(truelabel, rscore)


if __name__=="__main__":
    print 'argv ', sys.argv
    if len(sys.argv)>1:
        demoid = int(sys.argv[1])
    else:
        demoid = 0
    'number of block to find'
    if len(sys.argv)>2:
        K=int(sys.argv[2])
    else:
        K=1

    blnm = 'HS'
    respath = './testout/'
    if demoid == 0:
        print 'demo on sythetic data with hyperbolic block'
        rootnm='hycomm'
        ratefile, tsfile, tunit = None, None, None
        A1,B1,A2,B2= 500,500, 2500, 2500
        m = genDiHyperRectBlocks(A1, B1, A2, B2, alpha=-0.5, tau=0.002)
        m = addnosie(m, A1+A2, B1+B2, 0.01, black=True, A0=0, B0=0)
        m = addnosie(m, A1+A2, B1+B2, 0.4, black=False, A0=0, B0=0)
        sm = coo_matrix(m)
        acnt, bcnt = 500,500
        M = injectCliqueCamo(sm, acnt, bcnt, p=0.6, testIdx=3)
        trueA, trueB = range(acnt), range(bcnt)
        if True:
            import matplotlib.pyplot as plt
            fig=plt.figure()
            plt.spy(M, marker=',', markersize = 1)
            fig.show()
        alg = 'fastgreedy'#'greedy' #
        ptype=[Ptype.freq]
        qfun, b='exp', 128
    elif demoid == 1:
        print 'demo on real data with injected labels'
        rootnm = 'yelp'
        path = './testdata/'
        'tunit is used by HoloScope'
        tunit = 'd'
        freqfile = path+rootnm+'.edgelist'
        ratefile = path+rootnm+'rate.dict'
        tsfile = path+rootnm+'ts.dict'
        if not (os.path.isfile(freqfile+'.gz')):
            print 'file does not exists'
            sys.exit(1)
        if not os.path.isfile(ratefile+'.gz'):
            ratefile=None
        if not os.path.isfile(tsfile+'.gz'):
            tsfile=None
        'inject unpopular object with indegree at most 100'
        bcnt, goal, popbd = 200, 200, 100
        acnt = 2000

        print 'inject: \n\t{}\n\t{}\n\t{}'.format(freqfile, ratefile,
                                                  tsfile)
        print '\tinject: A:{} B:{} goal:{}'.format(acnt, bcnt, goal)
        suffix='.{}'.format(acnt)
        M, (trueA, trueB) = injectFraud2PropGraph( freqfile, ratefile, tsfile,
                                                  acnt, bcnt, goal = goal, popbd=popbd,
                                                  testIdx = 3, suffix=suffix)
        freqfile, ratefile, tsfile = freqfile+'.inject'+suffix,\
                                     ratefile+'.inject'+suffix,\
                                     tsfile+'.inject'+suffix
        alg = 'fastgreedy'
        ptype  = [Ptype.freq, Ptype.ts, Ptype.rate]
        qfun, b = 'exp', 32
    else:
        print 'no demo {}'.format(demoid)
        print 'try demo id 1 or 2'

    M = M.asfptype()
    bdres = HoloScope(M, alg, ptype, qfun=qfun, b=b,
                      ratefile=ratefile, tsfile=tsfile, tunit=tunit,
                      nblock=K)
    opt = bdres[-1]
    for nb in xrange(K):
        res = opt.nbests[nb]
        pr = precision_recall_sets(set(trueA), set(res[1][0]))
        FA = Fmeasure(pr[0],pr[1])
        print 'block{}: \n\tobjective value {}'.format(nb, res[0])
        print '\tA precision:{}, recall:{}, F:{}'\
                .format(pr[0],pr[1],FA)
        auc = auc_trueset_rankscore(trueB, res[1][1])
        print '\tB auc:{}'.format(auc)
        T = respath+rootnm+'a{}.blk{}'.format(acnt,nb)
        saveSimpleListData(res[1][0], T+'.rows')
        saveSimpleListData(res[1][1], T+'.colscores')
#end main

