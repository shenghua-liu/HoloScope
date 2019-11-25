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

def loadpreprocess(rootnm, path):
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
    infreqfile, inratefile, intsfile = freqfile+'.inject'+suffix,\
                                 ratefile+'.inject'+suffix,\
                                 tsfile+'.inject'+suffix
    ftA, ftB = freqfile +'.trueA'+suffix, \
            freqfile +'.trueB'+suffix
    if os.path.isfile(infreqfile) and \
        os.path.isfile(intsfile) and \
        os.path.isfile(inratefile):
        print 'reading existing injected files... ...'
    else:
        M, (trueA, trueB) = injectFraud2PropGraph( freqfile, ratefile, tsfile,
                acnt, bcnt, goal = goal, popbd=popbd, testIdx = 3, suffix=suffix)

    return infreqfile, intsfile, inratefile, ftA, ftB

def checkfilegz(name):
    if os.path.isfile(name):
        return name
    elif os.path.isfile(name+'.gz'):
        return name+'.gz'
    else:
        return None

def scoreLevelObjects( objscores ):
    sortscores = sorted(objscores, reverse=True)
    sortobjs = np.argsort(objscores)[::-1]
    diffscores = - np.diff(sortscores)
    levelid = np.argmax(diffscores)
    levelobjs = sortobjs[ : levelid+1]
    return levelobjs

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run",
            help="choose which way to run (default:0). \
            0 and 1 are demo run; 2 run for user's input case",
            choices=[0,1,2], type=int, default=0)
    parser.add_argument('-K', '--nblock',
            help="choose number of blocks to detect (default:1).",
            type=int, default=1)
    parser.add_argument('-p', '--outpath',
            help="choose number of blocks to detect (default:./testout/).",
            default='./testout/')
    parser.add_argument('path', nargs='?', default='./testdata/',
            help="the path of input data")
    parser.add_argument('rootnm', nargs='?', default='',
            help='''the root name of input files, including *.edgelist,
                    *ts.dict for time series, *rate.dict for rating.
                    e.g. yelp.edgelist[.gz], yelpts.dict[.gz],
                    yelprate.dict[.gz]''')
    parser.add_argument('-t', action='store_true',
            help='consider time series. Need *ts.dict[.gz] file')
    parser.add_argument('-s', action='store_true',
            help='consider rating scores. Need *rate.dict[.gz] file')
    args = parser.parse_args()

    respath =  args.outpath
    K = args.nblock
    if args.run == 0:
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
            plt.savefig(respath+'demo0.png')
            fig.show()
        alg = 'fastgreedy'#'greedy' #
        ptype=[Ptype.freq]
        qfun, b='exp', 128
    elif args.run == 1:
        print 'demo on real data with injected labels'
        rootnm = 'yelp'
        path = './testdata/'
        freqfile, tsfile, ratefile, ftA, ftB = loadpreprocess(rootnm, path)
        M = loadedge2sm(freqfile, coo_matrix, weighted=True)
        alg = 'fastgreedy'
        ptype  = [Ptype.freq, Ptype.ts, Ptype.rate] #[Ptype.freq] 
        qfun, b = 'exp', 32 #8 #10 #4 #8 # 32
        'tunit is used by HoloScope'
        tunit = 'd'
        print 'load ground truth ... ...'
        trueA = loadSimpleList(ftA, dtype=int)
        trueB = loadSimpleList(ftB, dtype=int)
    elif args.run == 2:
        print 'run users data and output into files'
        rootnm = args.rootnm
        path = args.path
        if rootnm == '':
            print 'please give root name, see help with -h or --help.'
            sys.exit()
        ptype=[Ptype.freq]
        freqfile, tsfile, ratefile = checkfilegz(path+rootnm+'.edgelist'), None, None
        if args.t:
            ptype.append(Ptype.ts)
            tsfile = checkfilegz(path+rootnm+'ts.dict')
        if args.s:
            ptype.append(Ptype.rate)
            ratefile = checkfilegz(path+rootnm+'rate.dict')
        M = loadedge2sm(freqfile, coo_matrix, weighted=True)
        alg = 'fastgreedy'
        qfun, b = 'exp', 32 #8 #10 #4 #8 # 32 
        tunit = 'd'
    else:
        #print 'no demo {}'.format(demoid)
        #print 'try demo id 0 (default) or 1'
        sys.exit()

    M = M.asfptype()
    bdres = HoloScope(M, alg, ptype, qfun=qfun, b=b,
                      ratefile=ratefile, tsfile=tsfile, tunit=tunit,
                      nblock=K)
    opt = bdres[-1]
    for nb in xrange(K):
        res = opt.nbests[nb]
        print 'block{}: \n\tobjective value {}'.format(nb+1, res[0])
        levelcols = scoreLevelObjects(res[1][1])
        if args.run < 2:
            pr = precision_recall_sets(set(trueA), set(res[1][0]))
            FA = Fmeasure(pr[0],pr[1])
            print '\tA precision:{}, recall:{}, F:{}'\
                    .format(pr[0],pr[1],FA)
            auc = auc_trueset_rankscore(trueB, res[1][1])
            print '\tB auc:{}'.format(auc)
            prB = precision_recall_sets(set(trueB), set(levelcols))
            FB = Fmeasure(prB[0], prB[1])
            print '\tB precision:{}, recall:{}, F:{}'\
                    .format(prB[0], prB[1], FB)

        T = respath+rootnm+'.blk{}'.format(nb+1)
        saveSimpleListData(res[1][0], T+'.rows')
        saveSimpleListData(res[1][1], T+'.colscores')
        saveSimpleListData(levelcols, T+'.levelcols')
#end main

