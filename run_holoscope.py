import sys
from gendenseblock import *
import numpy as np
import contextlib
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


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = basicio.BytesIO()
    yield
    sys.stdout = save_stdout
    return

if __name__=="__main__":

    ptype  = [Ptype.freq, Ptype.ts, Ptype.rate]
    sdrop = True
    aggmethod = 'sum' #'rank'
    qfun, b = 'exp', 32

    path = './testdata/'
    respath = './testout/results/'
    rootnm = 'yelp'
    'tunit is used by HoloScope'
    tunit = 'd'
    tbins = 1 #one day
    blnm = 'HS'
    prAsdic, FAsdic, aucBsdic = \
            {rootnm+blnm:[]}, \
            {rootnm+blnm:[]}, \
            {rootnm+blnm:[]}

    freqfile = path+rootnm+'.edgelist'
    ratefile = path+rootnm+'rate.dict'
    tsfile = path+rootnm+'ts.dict'
    if not (os.path.isfile(freqfile)):
        print 'file does not exists'
        sys.exit(1)
    if not os.path.isfile(ratefile):
        ratefile=None
    if not os.path.isfile(tsfile):
        tsfile=None
    bcnt = 200
    goal = 200
    acnt = 4000
    popbd=100 #inject unpopular object with indegree at most 100

    print 'inject: \n\t{}\n\t{}\n\t{}'.format(freqfile, ratefile,
                                              tsfile)
    print '\tinject: A:{} B:{} goal:{}'.format(acnt, bcnt, goal)
    'attack target 100 prod, rate 100'
    suffix='.{}'.format(acnt)
    M, (trueA, trueB) = injectFraud2PropGraph(
        freqfile, ratefile, tsfile,
        acnt, bcnt, goal = goal, popbd=popbd,
        testIdx = 3, suffix=suffix)

    freqfile, ratefile, tsfile = freqfile+'.inject'+suffix,\
                                 ratefile+'.inject'+suffix,\
                                 tsfile+'.inject'+suffix
    M = M.asfptype()
    bdres = HoloScope(M, 'fastgreedy', ptype=ptype,
                      qfun=qfun, b=b, epsilon=0.0,
                      aggmethod=aggmethod, sdrop=sdrop,
                      tsfile=tsfile, tunit=tunit, ratefile=ratefile,
                      numSing=10, rbd='avg', nblock=1)

    for nb in xrange(nblock):
        kk=rootnm+blnm+str(nb)
        opt = bdres[-1]
        res = opt.nbests[nb]
        pr = precision_recall_sets(set(trueA), set(res[1][0]))
        FA = Fmeasure(pr[0],pr[1])
        print 'block{}: \n\tobjective value {}'.format(res[0])
        print '\tA precision:{}, recall:{}, F:{}'\
                .format(pr[0],pr[1],FA)
        auc = auc_trueset_rankscore(trueB, res[1][1])
        print '\tB auc:{}'.format(auc)
        T = respath+rootnm+'b{}.blk{}'.format(bcnt,nb)
        saveSimpleListData(res[1][0], T+'.rows')
        saveSimpleListData(res[1][1], T+'.colscores')
    # rootnm loop
    #end test paths loop
#end if main

